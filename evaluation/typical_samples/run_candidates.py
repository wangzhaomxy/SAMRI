#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run ONE model on every slice of typical_samples_all_models.csv with the benchmark's preprocessing and prompt,
and save its masks and Dice. run_all_models.sh calls it once per model, each time in that model's own conda
environment (/scratch/user/s4670484/conda/<env>).

    model        environment                     code (benchmark protocol)
    sam_box      sam_med2d (CUDA)                typical_samples.predict_sam: list box
    samri_box    sam_med2d (CUDA)                typical_samples.predict_sam: list box
    samri_bp     sam_med2d (CUDA)                typical_samples.predict_sam: list box + list point
    medsam       sam_med2d (CUDA)                typical_samples.predict_medsam (needs scikit-image)
    sam_med2d    sam_med2d (CUDA)                test_comparison_models.infer_sam_med2d: own box, 256 frame
    mcp_medsam   mcp-medsam (CUDA)               test_comparison_models.infer_mcp_medsam: list box
    samed        samed (CUDA, Python 3.8)        test_comparison_models.infer_samed: no prompt

Output in --out:  masks_<model>.npz (one boolean mask per sample_id) and dice_<model>.csv (Dice of this run,
Dice in the benchmark records, Dice of the v2 run where there is one). Python 3.8 compatible.

  python run_candidates.py <model> --list <typical_samples_all_models.csv> --out <folder> [--device cpu] [--limit N]
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

REPO = "/scratch/user/s4670484/SAMRI"
PRE = "/scratch/user/s4670484/pretrained_ckpt"
CMP = "/scratch/user/s4670484/comparison_ckpt_samri"
CKPT = {"sam_box": PRE + "/sam_vit_b_01ec64.pth", "samri_box": PRE + "/samri_vitb_box.pth",
        "samri_bp": PRE + "/samri_vitb_bp.pth", "medsam": PRE + "/medsam_vit_b.pth",
        "sam_med2d": CMP + "/SAM_Med2D/sam-med2d_b.pth", "mcp_medsam": CMP + "/mcp_medsam/mcp_best.pth",
        "samed": CMP + "/SAMed/sam_vit_b_01ec64.pth"}
SAMED_LORA, SAMED_RANK = CMP + "/SAMed/epoch_159.pth", 4
TOL = 0.005
BOX = ("box_x0", "box_y0", "box_x1", "box_y1")


def num(x):
    return None if x in (None, "") else float(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=sorted(CKPT))
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="first N slices only (quick test)")
    ap.add_argument("--repo-root", default=REPO)
    args = ap.parse_args()
    m, dev, t0 = args.model, args.device, time.time()
    for f in [CKPT[m]] + ([SAMED_LORA] if m == "samed" else []):
        if not os.path.isfile(f):
            sys.exit("checkpoint not found: " + f)

    sys.path[:0] = [args.repo_root, os.path.join(args.repo_root, "evaluation"),
                    os.path.join(args.repo_root, "evaluation", "typical_samples")]
    import torch
    from typical_samples import load_slice, sam_input, dice
    if dev.startswith("cuda") and not torch.cuda.is_available():
        sys.exit("[%s] no CUDA GPU visible (python %s, torch %s)" % (m, sys.executable, torch.__version__))
    print("[%s] %s | torch %s | %s" % (m, sys.executable, torch.__version__,
          torch.cuda.get_device_name(0) if dev.startswith("cuda") else "CPU, %d threads" % torch.get_num_threads()),
          flush=True)

    rows = list(csv.DictReader(open(args.list, newline="")))
    if args.limit:
        rows = rows[:args.limit]
    for r in rows:
        r["key"], r["label_value"] = r["sample_id"], float(r["label_value"])
    slices = dict((r["key"], load_slice(r)) for r in rows)          # (raw slice, structure mask)

    if m in ("sam_box", "samri_box", "samri_bp", "medsam"):
        from typical_samples import load_model, predict_sam, predict_medsam
        net = load_model(CKPT[m], dev)
        if m == "medsam":
            pred = predict_medsam(net, rows, slices, dev)
        else:
            pred = predict_sam(net, rows, slices, with_point=(m == "samri_bp"), seed=0)
        masks = dict((k, v[0]) for k, v in pred.items())
        off = 0                                                     # the prompts must be the list's
        for r in rows:
            box, point = pred[r["key"]][1], pred[r["key"]][2]
            off += [int(round(float(v))) for v in box] != [int(r[c]) for c in BOX]
            if point is not None:
                off += [int(v) for v in point] != [int(r["point_x"]), int(r["point_y"])]
        print("[%s] prompts identical to the list for %d of %d slices" % (m, len(rows) - off, len(rows)), flush=True)
    else:
        import test_comparison_models as cmp
        if m == "mcp_medsam":
            net, clip, tok = cmp.load_mcp_medsam(CKPT[m], dev)
            run = lambda r, img, gt: cmp.infer_mcp_medsam(net, clip, tok, img,
                                                          np.array([int(r[c]) for c in BOX]), dev)
        elif m == "sam_med2d":
            net = cmp.load_sam_med2d(CKPT[m], dev)
            run = lambda r, img, gt: cmp.infer_sam_med2d(net, img, gt, dev)
        else:
            net = cmp.load_samed(CKPT[m], SAMED_LORA, SAMED_RANK, dev)
            run = lambda r, img, gt: cmp.infer_samed(net, img, dev, int(r["label_value"]))
        masks = {}
        for r in rows:
            img, gt = slices[r["key"]]
            masks[r["key"]] = np.asarray(run(r, sam_input(img), gt)).astype(bool)

    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, "masks_%s.npz" % m), **masks)
    rec_d, v2_d, ds = [], [], []
    with open(os.path.join(args.out, "dice_%s.csv" % m), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "model", "dice", "record", "v2"])
        for r in rows:
            d = dice(masks[r["key"]], slices[r["key"]][1])
            rec, v2 = num(r.get("rec_" + m)), num(r.get("v2_" + m))
            w.writerow([r["sample_id"], m, round(d, 4), "" if rec is None else rec, "" if v2 is None else v2])
            ds.append(d)
            if rec is not None:
                rec_d.append(abs(d - rec))
            if v2 is not None:
                v2_d.append(abs(d - v2))
    line = "[%s] %d slices, mean Dice %.4f; benchmark records: %d of %d within %.3f (median |diff| %.4f)" % (
        m, len(ds), float(np.mean(ds)), sum(x <= TOL for x in rec_d), len(rec_d), TOL, float(np.median(rec_d)))
    if v2_d:
        line += "; v2 run: %d of %d within %.3f" % (sum(x <= TOL for x in v2_d), len(v2_d), TOL)
    print(line + "; %.0f s -> %s" % (time.time() - t0, args.out), flush=True)


if __name__ == "__main__":
    main()
