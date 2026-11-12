#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nnU-Net comparison, step 1 (CPU): one 2D nnU-Net dataset per training dataset, plus its test images.

Datasets: the 11 clean datasets and MSK_T2, each a separate specialist nnU-Net trained on its own data:
  training   every slice of SAMRI_train_test/<dataset>/training
  validation every slice of SAMRI_train_test/<dataset>/validation (the patient-wise split), used as nnU-Net's
             fold 0 validation set through splits_final.json; a dataset without validation slices trains on
             fold "all"
Test images (the slices every other system is scored on):
  the 11 clean datasets  their in-domain test rows (in-domain manifest)          test set "in-domain"
  MSK_T2                 every test slice of MSK_T2, MSK_PD, MSK_FLASH (knee manifest)   test set "knee"
nnU-Net cannot segment classes it was not trained on, so no zero-shot set is tested; the only test outside a
model's own dataset is the MSK_T2 model on the other knee sequences, which have the same six classes.

Each slice is written as a 2D NIfTI with SimpleITK, which nnU-Net reads as a 2D image. Label values are
mapped to 1..K for nnU-Net (samri_map.json keeps the original values; nnunet_run.py maps them back).

Lesion datasets (ISLES2022_ADC, ISLES2022_DWI, QIN-PROSTATE-Lesion) store one label number per lesion (2, 3, 4,
... up to 109: the lesion's number within its scan). The number is not a class. Mapping each number to its own
class (array indices 4, 5 and 8, the first run) asks nnU-Net to tell lesion 2 from lesion 3 by appearance,
which it cannot learn: in that run it found almost no lesion numbered 3 or higher. These three are therefore
written a second time as one-label datasets ("onelabel": every lesion -> class 1), after the 12 above: array
indices 12-14, dataset ids first-id + 12..14. nnunet_run.py scores them per lesion (each predicted lesion pixel
goes to the nearest true lesion) and marks the first-run results of the same datasets as superseded.

Outputs under --nnunet-root: raw/DatasetNNN_<name>/ (imagesTr, labelsTr, dataset.json, samri_map.json,
splits_final.json), test/DatasetNNN_<name>/ (imagesTs, index.json), datasets.json. A dataset with a DONE file
is skipped.
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emb_common as ec  # noqa: E402

DATASETS = ["BraTS_FLAIR", "BraTS_T1", "BraTS_T1CE", "Heart", "ISLES2022_ADC", "ISLES2022_DWI", "MSK_T2",
            "ProstateT2", "QIN-PROSTATE-Lesion", "QIN-PROSTATE-Prostate", "SpineMR", "crossmoda"]
KNEE_TEST = ("MSK_T2", "MSK_PD", "MSK_FLASH")
ONE_LABEL = ("ISLES2022_ADC", "ISLES2022_DWI", "QIN-PROSTATE-Lesion")  # label value = lesion number, not class


def _load_seg(seg_path):
    import nibabel as nib
    return np.rint(np.asarray(nib.load(seg_path).get_fdata())[0]).astype(np.int64)


def _scan_seg(seg_path):
    seg = _load_seg(seg_path)
    vals, counts = np.unique(seg, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts) if v != 0}, seg.shape


def _write_image(args):
    import SimpleITK as sitk
    img_path, out = args
    raw = ec.load_slice(img_path).astype(np.float32)
    sitk.WriteImage(sitk.GetImageFromArray(raw), out)
    return raw.shape


def _write_label(args):
    import SimpleITK as sitk
    seg_path, out, mapping = args
    seg = _load_seg(seg_path)
    lut = np.zeros(max(max(mapping), int(seg.max())) + 1, dtype=np.uint8)
    for orig, new in mapping.items():
        lut[orig] = new
    sitk.WriteImage(sitk.GetImageFromArray(lut[seg]), out)


def folder_name(i, ds, first_id):
    return f"Dataset{first_id + i:03d}_{re.sub(r'[^A-Za-z0-9]+', '_', ds)}"


def label_stats(pool, seg_paths):
    area, count = defaultdict(int), defaultdict(int)
    for vals, _ in pool.map(_scan_seg, seg_paths, chunksize=64):
        for v, c in vals.items():
            area[v] += c
            count[v] += 1
    return area, count


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traintest", required=True, help="Datasets/SAMRI_train_test")
    ap.add_argument("--indomain-dir", required=True, help="embedding_research_clean (in-domain test rows)")
    ap.add_argument("--knee-dir", required=True, help="embedding_research_knee (MSK_T2/PD/FLASH test rows)")
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--nnunet-root", required=True)
    ap.add_argument("--first-id", type=int, default=901)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    t0 = time.time()
    from build_manifest import list_pairs

    names = sorted(args.datasets)
    # ---- test rows: the manifests every other system is scored from -------------------------------------
    tests = defaultdict(list)
    for wdir, keep, label in ((args.indomain_dir, None, "in-domain"), (args.knee_dir, KNEE_TEST, "knee")):
        path = os.path.join(wdir, "manifest.csv")
        if not os.path.isfile(path):
            sys.exit(f"missing {path}" + (" - run step 1 of run_knee_experiments.sh first" if label == "knee" else ""))
        for r in ec.read_manifest(path):
            if r["split"] != "test":
                continue
            if keep is None and r["dataset"] in names:
                tests[r["dataset"]].append((label, r))
            elif keep is not None and r["dataset"] in keep and "MSK_T2" in names:
                tests["MSK_T2"].append((label, r))

    os.makedirs(args.nnunet_root, exist_ok=True)
    # the 12 datasets keep their array indices (0-11, sorted by name); the one-label variants come after them
    jobs = [(ds, None) for ds in names] + [(ds, "onelabel") for ds in ONE_LABEL if ds in names]
    summary = []
    with Pool(args.workers) as pool:
        for i, (ds, variant) in enumerate(jobs):
            fname = folder_name(i, ds + (f"_{variant}" if variant else ""), args.first_id)
            raw = os.path.join(args.nnunet_root, "raw", fname)
            tdir = os.path.join(args.nnunet_root, "test", fname)
            done = os.path.join(raw, "DONE")
            if os.path.isfile(done):
                summary.append(json.load(open(done)))
                print(f"[{i}] {fname}: done - skipping")
                continue
            t1 = time.time()
            splits = {}
            for sub in ("training", "validation"):
                folder = os.path.join(args.traintest, ds, sub)
                splits[sub] = list_pairs(folder)[0] if os.path.isdir(folder) else []
            if not splits["training"]:
                sys.exit(f"no training slices in {os.path.join(args.traintest, ds, 'training')}")
            tag = ds.replace("-", "_")
            cases = {sub: [f"{tag}_{sub[:2]}_{j:06d}" for j in range(len(splits[sub]))] for sub in splits}
            pairs = splits["training"] + splits["validation"]
            all_cases = cases["training"] + cases["validation"]
            for sub in ("imagesTr", "labelsTr"):
                os.makedirs(os.path.join(raw, sub), exist_ok=True)
            os.makedirs(os.path.join(tdir, "imagesTs"), exist_ok=True)

            area, count = label_stats(pool, [s for _, s in pairs])
            if variant == "onelabel":
                mapping = {v: 1 for v in sorted(area)}
            else:
                mapping = {v: k + 1 for k, v in enumerate(sorted(area))}
            fold = "0" if splits["validation"] else "all"
            print(f"[{i}] {fname}: {len(splits['training'])} training + {len(splits['validation'])} validation slices "
                  f"(fold {fold}){'; one label for every lesion' if variant else ''}; "
                  f"labels (original -> nnU-Net; slices; mean area): "
                  + ", ".join(f"{v}->{mapping[v]} ({count[v]}; {area[v] / count[v]:.0f})" for v in sorted(area)),
                  flush=True)

            shapes = pool.map(_write_image, [(img, os.path.join(raw, "imagesTr", c + "_0000.nii.gz"))
                                             for c, (img, _) in zip(all_cases, pairs)], chunksize=64)
            seg_shapes = pool.map(_scan_seg, [s for _, s in pairs], chunksize=64)
            bad = [c for c, s, (_, sh) in zip(all_cases, shapes, seg_shapes) if s != sh]
            if bad:
                sys.exit(f"image and mask shapes differ for {len(bad)} slices, e.g. {bad[0]}")
            pool.map(_write_label, [(seg, os.path.join(raw, "labelsTr", c + ".nii.gz"), mapping)
                                    for c, (_, seg) in zip(all_cases, pairs)], chunksize=64)
            classes = ({"lesion": 1} if variant == "onelabel"
                       else {f"label_{v}": mapping[v] for v in sorted(mapping)})
            json.dump({"channel_names": {"0": "MRI"}, "labels": {"background": 0, **classes},
                       "numTraining": len(all_cases), "file_ending": ".nii.gz",
                       "description": f"SAMRI {ds}, 2D slices; original label values in samri_map.json"},
                      open(os.path.join(raw, "dataset.json"), "w"), indent=1)
            if fold == "0":
                json.dump([{"train": cases["training"], "val": cases["validation"]}],
                          open(os.path.join(raw, "splits_final.json"), "w"))
            json.dump({"dataset": ds, "fold": fold, "variant": variant, "onelabel": variant == "onelabel",
                       # one-label: class 1 = any lesion; the original values are lesion numbers (scored per lesion)
                       "labels": ({"1": "lesion"} if variant == "onelabel"
                                  else {str(n): v for v, n in mapping.items()}),
                       "lesion_numbers": sorted(mapping) if variant == "onelabel" else None,
                       "cases": dict(zip(all_cases, (p[0] for p in pairs)))},
                      open(os.path.join(raw, "samri_map.json"), "w"))

            # test images
            index = [{"case": f"{ts.replace('-', '')}_{r['row']:06d}", "test_set": ts, "root": r["root"],
                      "dataset": r["dataset"], "row": r["row"], "img_path": r["img_path"], "seg_path": r["seg_path"]}
                     for ts, r in tests[ds]]
            pool.map(_write_image, [(e["img_path"], os.path.join(tdir, "imagesTs", e["case"] + "_0000.nii.gz"))
                                    for e in index], chunksize=16)
            json.dump(index, open(os.path.join(tdir, "index.json"), "w"))
            by = defaultdict(int)
            for e in index:
                by[(e["test_set"], e["dataset"])] += 1
            print("      test slices: " + ", ".join(f"{d} ({t}) {n}" for (t, d), n in sorted(by.items())))
            if ds == "MSK_T2" and not variant:  # do the other sets' label values mean the same structures? compare mean areas
                for other in ("MSK_PD", "MSK_FLASH"):
                    oa, oc = label_stats(pool, [e["seg_path"] for e in index if e["dataset"] == other])
                    print(f"      labels {other} test (label: slices; mean area; MSK_T2 mean area): "
                          + ", ".join(f"{v}: {oc[v]}; {oa[v] / oc[v]:.0f}; "
                                      + (f"{area[v] / count[v]:.0f}" if v in area else "not an MSK_T2 class")
                                      for v in sorted(oa)))
            entry = {"index": i, "id": args.first_id + i, "name": ds, "folder": fname, "fold": fold,
                     "n_train": len(splits["training"]), "n_val": len(splits["validation"]), "n_test": len(index)}
            if variant:
                entry["variant"] = variant
            summary.append(entry)
            json.dump(entry, open(done, "w"))
            print(f"      {time.time() - t1:.0f} s", flush=True)
    json.dump(summary, open(os.path.join(args.nnunet_root, "datasets.json"), "w"), indent=1)
    print(f"{len(summary)} datasets -> {args.nnunet_root} ({time.time() - t0:.0f} s)")
    for e in summary:
        print(f"  array index {e['index']:2d}: {e['folder']} ({e['n_train']} train, {e['n_val']} validation, "
              f"{e['n_test']} test; fold {e['fold']})"
              + (" - superseded by its one-label version" if any(x.get("variant") == "onelabel" and x["name"] == e["name"]
                                                                  for x in summary) and not e.get("variant") else ""))


if __name__ == "__main__":
    main()
