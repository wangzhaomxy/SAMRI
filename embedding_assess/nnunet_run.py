#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nnU-Net comparison, step 2 (1 GPU per dataset): train one 2D nnU-Net, predict its test slices and score
them the way eval_decoder.py does (one instance per label value present in a slice; Dice at native size).

Training is nnU-Net's own: default plans and trainer, 2D, fold 0 = the dataset's own patient-wise
training/validation split (splits_final.json from nnunet_prepare.py), then nnU-Net's final validation on the
validation slices (its summary.json is a sanity check, not a result). Prediction uses checkpoint_final.pth.
Resumable: an unfinished run continues from checkpoint_latest.pth (saved every 50 epochs), predictions already
written are kept, and a finished run goes straight to prediction.
The MSK_T2 model is also tested on MSK_PD and MSK_FLASH (same six classes, other sequences); no model is
tested on the zero-shot sets, whose classes it was not trained on.
One-label lesion datasets (nnunet_prepare.py: ISLES2022_ADC/DWI and QIN-PROSTATE-Lesion, every lesion = class 1)
are scored per lesion, like the box-prompted systems: each true lesion (label number) in a slice is one
structure, and each predicted lesion pixel is given to the nearest true lesion of that slice (so a false
positive counts against the lesion it lies closest to). The rows keep the original label numbers, so they pair
with eval_decoder.py's rows.
Output: <nnunet-root>/eval/<trainer>/eval_instances_<DatasetNNN_name>.csv, with the columns of
eval_decoder.py's eval_instances.csv (merge with analyse_knee.py --instances). For a dataset that also has a
one-label version, the first-run file (one class per lesion number) is renamed to *.csv.superseded, so a
*.csv glob picks up exactly one nnU-Net result per test structure.
"""

import argparse
import csv
import json
import os
import shutil
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emb_common as ec  # noqa: E402


def entry_key(e):
    return e["name"] + (f"_{e['variant']}" if e.get("variant") else "")


def lesion_dice(pred_bin, seg):
    """Per-lesion Dice when the model predicts one lesion class: every pixel goes to its nearest true lesion
    (nearest-neighbour partition of the slice), and lesion l is scored on the predicted pixels in its part."""
    from scipy import ndimage
    labs = ec.slice_labels(seg)
    if len(labs) == 1:
        return {labs[0]: ec.dice(pred_bin, seg == labs[0])}
    idx = ndimage.distance_transform_edt(seg == 0, return_distances=False, return_indices=True)
    owner = seg[tuple(idx)]
    return {lab: ec.dice(pred_bin & (owner == lab), seg == lab) for lab in labs}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nnunet-root", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--index", type=int, help="array index (see datasets.json)")
    g.add_argument("--dataset", help="dataset name, e.g. MSK_T2 or ISLES2022_ADC_onelabel")
    ap.add_argument("--trainer", default="nnUNetTrainer", help="nnUNetTrainer_5epochs for a smoke test")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    t0 = time.time()

    root = args.nnunet_root
    for k, sub in (("nnUNet_raw", "raw"), ("nnUNet_preprocessed", "preprocessed"), ("nnUNet_results", "results")):
        os.environ[k] = os.path.join(root, sub)
    os.environ.setdefault("nnUNet_n_proc_DA", str(args.workers))
    import torch
    from torch.backends import cudnn

    datasets = json.load(open(os.path.join(root, "datasets.json")))
    entry = next((e for e in datasets if (e["index"] == args.index if args.index is not None
                                          else entry_key(e) == args.dataset)), None)
    if entry is None:
        sys.exit(f"no dataset {args.index if args.index is not None else args.dataset} in datasets.json")
    folder, ds, fold = entry["folder"], entry["name"], entry["fold"]
    fold_arg = fold if fold == "all" else int(fold)
    print(f"{folder}: {entry['n_train']} training, {entry['n_val']} validation, {entry['n_test']} test slices; "
          f"fold {fold}; trainer {args.trainer}")
    print(f"python {sys.executable}  torch {torch.__version__}  hip {torch.version.hip}")
    device = torch.device(args.device)
    if device.type == "cuda":
        assert torch.cuda.is_available(), "no GPU visible - wrong conda env for this card?"
        print("device:", torch.cuda.get_device_name(0))
    model_dir = os.path.join(root, "results", folder, f"{args.trainer}__nnUNetPlans__2d")
    fold_dir = os.path.join(model_dir, f"fold_{fold}")
    final = os.path.join(fold_dir, "checkpoint_final.pth")
    val_summary = os.path.join(fold_dir, "validation", "summary.json")

    # ---- train (+ nnU-Net's final validation on the validation split) ----------------------------------------
    if fold != "all":  # our patient-wise split, not nnU-Net's random 5-fold split of slices
        src = os.path.join(root, "raw", folder, "splits_final.json")
        dst = os.path.join(root, "preprocessed", folder, "splits_final.json")
        if not os.path.isfile(dst) or open(src).read() != open(dst).read():
            if os.path.isfile(os.path.join(fold_dir, "checkpoint_latest.pth")):
                sys.exit(f"{dst} differs from {src} but a training has started - stopping")
            shutil.copyfile(src, dst)
    if os.path.isfile(final) and (fold == "all" or os.path.isfile(val_summary)):
        print("training: done - skipping")
    else:
        from nnunetv2.run.run_training import get_trainer_from_args, maybe_load_checkpoint
        trainer = get_trainer_from_args(str(entry["id"]), "2d", fold_arg, args.trainer, "nnUNetPlans",
                                        continue_training=True, device=device)
        maybe_load_checkpoint(trainer, True, False)
        if device.type == "cuda":
            cudnn.deterministic, cudnn.benchmark = False, True
        print(f"training from epoch {trainer.current_epoch} of {trainer.num_epochs}", flush=True)
        trainer.run_training()
        log = getattr(trainer.logger, "local_logger", trainer.logger).my_fantastic_logging  # 2.7: MetaLogger
        dur = np.array(log["epoch_end_timestamps"]) - np.array(log["epoch_start_timestamps"])
        if len(dur) > 1:
            med = float(np.median(dur[1:]))
            print(f"epoch time: median {med:.1f} s over epochs 2-{len(dur)} (first {dur[0]:.0f} s, MIOpen tuning) "
                  f"-> 1000 epochs = {med * 1000 / 3600:.1f} h", flush=True)
        print(f"training done ({(time.time() - t0) / 3600:.2f} h)", flush=True)
        if fold != "all":
            t1 = time.time()
            trainer.perform_actual_validation(False)
            print(f"nnU-Net validation on {entry['n_val']} validation slices ({(time.time() - t1) / 60:.1f} min)")
    if os.path.isfile(val_summary):
        fm = json.load(open(val_summary)).get("foreground_mean", {})
        print(f"nnU-Net's own validation (patient-wise held-out slices): foreground mean Dice {fm.get('Dice', float('nan')):.3f}")

    # ---- predict ---------------------------------------------------------------------------------------------
    tdir = os.path.join(root, "test", folder)
    index = json.load(open(os.path.join(tdir, "index.json")))
    pred_dir = os.path.join(root, "pred", args.trainer, folder)
    t1 = time.time()
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                perform_everything_on_device=device.type == "cuda", device=device, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(model_dir, use_folds=(fold_arg,), checkpoint_name="checkpoint_final.pth")
    predictor.predict_from_files(os.path.join(tdir, "imagesTs"), pred_dir, save_probabilities=False, overwrite=False,
                                 num_processes_preprocessing=max(1, args.workers // 2),
                                 num_processes_segmentation_export=max(1, args.workers // 2))
    print(f"prediction: {len(index)} test slices ({time.time() - t1:.0f} s)", flush=True)

    # ---- score -------------------------------------------------------------------------------------------------
    import SimpleITK as sitk
    smap = json.load(open(os.path.join(root, "raw", folder, "samri_map.json")))
    onelabel = bool(smap.get("onelabel"))
    if onelabel:
        print("scoring: one lesion class, per lesion (predicted pixels go to the nearest true lesion)")
    else:
        inv = {int(n): float(v) for n, v in smap["labels"].items()}  # nnU-Net label -> original label value
        lut = np.zeros(max(inv) + 1, dtype=np.float64)
        for n, v in inv.items():
            lut[n] = v
    system = "nnunet-2d" + ("" if args.trainer == "nnUNetTrainer" else f"_{args.trainer}")
    name = "nnU-Net 2D, one model per dataset, trained on that dataset" + (
        "" if args.trainer == "nnUNetTrainer" else f" ({args.trainer})")
    recs, unknown, slice_dice = [], defaultdict(int), []
    for e in index:
        raw_pred = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(pred_dir, e["case"] + ".nii.gz"))).astype(np.int64)
        seg = ec.load_slice(e["img_path"], e["seg_path"])[1]
        if raw_pred.shape != seg.shape:
            sys.exit(f"{e['case']}: prediction {raw_pred.shape} vs label map {seg.shape}")
        if onelabel:
            scores = lesion_dice(raw_pred > 0, seg)
            slice_dice.append(ec.dice(raw_pred > 0, seg > 0))
        else:
            pred = lut[raw_pred]
            scores = {lab: ec.dice(pred == lab, seg == lab) for lab in ec.slice_labels(seg)}
        for lab, d in scores.items():
            if not onelabel and float(lab) not in inv.values():
                unknown[(e["dataset"], float(lab))] += 1
            recs.append({"system": system, "name": name, "seed": 0, "test_set": e["test_set"], "root": e["root"],
                         "dataset": e["dataset"], "row": e["row"], "label": float(lab), "area": int((seg == lab).sum()),
                         "dice": d})
    for (d, lab), n in sorted(unknown.items()):
        print(f"  WARNING: {d} label {lab:g} ({n} structures) is not a class of this model - scored as missed")
    out_dir = os.path.join(root, "eval", args.trainer)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"eval_instances_{folder}.csv")
    has_onelabel = any(x.get("variant") == "onelabel" and x["name"] == ds for x in datasets)
    if has_onelabel and not entry.get("variant"):  # first-run lesion model: keep its result out of *.csv globs
        out += ".superseded"
        print(f"  {ds} has a one-label version: this result is written as superseded")
    if onelabel:
        for x in datasets:
            old = os.path.join(out_dir, f"eval_instances_{x['folder']}.csv")
            if x["name"] == ds and not x.get("variant") and os.path.isfile(old):
                os.replace(old, old + ".superseded")
                print(f"  first-run result (one class per lesion number) -> {os.path.basename(old)}.superseded")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    by = defaultdict(list)
    for r in recs:
        by[(r["test_set"], r["dataset"])].append(r["dice"])
    for (t, d), v in sorted(by.items()):
        print(f"  {d:24s} ({t:9s}): median Dice {np.nanmedian(v):.3f} over {len(v)} structures")
    if onelabel:
        low = min(smap["lesion_numbers"])
        first = [r["dice"] for r in recs if r["label"] == low]
        later = [r["dice"] for r in recs if r["label"] != low]
        med = lambda v: f"{np.nanmedian(v):.3f}" if v else "-"  # noqa: E731
        print(f"  per lesion: label {low:g} median {med(first)} ({len(first)}), higher labels median {med(later)} "
              f"({len(later)}); misses (Dice < 0.01) {np.mean([r['dice'] < 0.01 for r in recs]):.0%}")
        print(f"  whole slice (all lesions as one mask): median Dice {med(slice_dice)} over {len(slice_dice)} slices")
    print(f"-> {out} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
