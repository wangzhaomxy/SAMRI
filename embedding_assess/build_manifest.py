#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1 of the embedding experiments (CPU, a few minutes).

Draws ONE fixed sample of 2D slices that every encoder is run on, and records,
for each pipeline ("sam" pad / "medsam" stretch), where every labelled
structure falls on the 64x64 token grid.

  train rows  <trained-root>/<dataset>/training/   (probe training only)
  test rows   <trained-root>/<dataset>/testing/    (probe testing + CKA)
              <zeroshot-root>/<dataset>/testing/   (zeroshot=1; never trained on)

Outputs in --work-dir:
  manifest.csv            row, root, dataset, split, zeroshot, img_path,
                          seg_path, H, W, n_labels
  geometry_sam.npz        per instance (slice x label): row, label, token
  geometry_medsam.npz     coverage (64x64, uint8 0-255 = fraction covered),
                          native box, box in token units; per row: valid-token
                          map and geometry (H, W, h, w)

Usage (CPU node, samri env, from the repo root):
  python embedding_assess/build_manifest.py \
      --trained-root /scratch/user/s4670484/Datasets/SAMRI_train_test/ \
      --zeroshot-root /scratch/user/s4670484/Datasets/Zeroshot/ \
      --work-dir /scratch/user/s4670484/Eval_results/embedding_research/
"""

import argparse
import csv
import os
import random
import sys
import time
from glob import glob

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emb_common as ec  # noqa: E402


def list_pairs(folder):
    """(image, mask) pairs matched by file name, so one missing file does not
    shift every later pair (sorted-order pairing would)."""
    imgs = sorted(glob(os.path.join(folder, f"*{ec.IMAGE_KEY}*")))
    segs = set(glob(os.path.join(folder, f"*{ec.MASK_KEY}*")))
    pairs, problems = [], []
    for img in imgs:
        seg = os.path.join(folder, os.path.basename(img).replace(ec.IMAGE_KEY, ec.MASK_KEY))
        if seg in segs:
            pairs.append((img, seg))
    no_mask = [os.path.basename(i) for i in imgs if i not in {p[0] for p in pairs}]
    no_img = sorted(os.path.basename(s) for s in segs - {p[1] for p in pairs})
    if no_mask:
        problems.append(f"{folder}: {len(no_mask)} images without a mask (skipped), e.g. {no_mask[0]}")
    if no_img:
        problems.append(f"{folder}: {len(no_img)} masks without an image (skipped), e.g. {no_img[0]}")
    return pairs, problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trained-root", default=None,
                    help="SAMRI_train_test/ (omit for a zero-shot-only, test-only manifest)")
    ap.add_argument("--zeroshot-root", default=None)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--n-train", type=int, default=50, help="training slices per trained dataset")
    ap.add_argument("--n-test", type=int, default=25, help="testing slices per dataset (0 = all)")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="only these dataset folder names (e.g. BraTS_FLAIR Heart); default all")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.work_dir, exist_ok=True)
    rng = random.Random(args.seed)
    plan, problems = [], []  # (root_name, dataset, split, zeroshot, img, seg)
    if not args.trained_root and not args.zeroshot_root:
        sys.exit("give --trained-root and/or --zeroshot-root")
    roots = ([(args.trained_root, 0)] if args.trained_root else []) + ([(args.zeroshot_root, 1)] if args.zeroshot_root else [])
    for root, zs in roots:
        root_name = os.path.basename(root.rstrip("/"))
        for ds in sorted(d for d in glob(os.path.join(root, "*")) if os.path.isdir(d)):
            name = os.path.basename(ds.rstrip("/"))
            if args.datasets and name not in args.datasets:
                continue
            splits = [("test", "testing", args.n_test)] + ([] if zs else [("train", "training", args.n_train)])
            for split, sub, n in splits:
                folder = os.path.join(ds, sub)
                if not os.path.isdir(folder):
                    continue
                pairs, p = list_pairs(folder)
                problems += p
                for img, seg in sorted(pairs if n == 0 else rng.sample(pairs, min(n, len(pairs)))):
                    plan.append((root_name, name, split, zs, img, seg))
    if args.datasets:
        missing = set(args.datasets) - {p[1] for p in plan}
        if missing:
            print("  WARNING: requested datasets not found:", ", ".join(sorted(missing)))
    print(f"Sampled {len(plan)} slices "
          f"({sum(p[2] == 'train' for p in plan)} train, {sum(p[2] == 'test' for p in plan)} test, "
          f"{len({(p[0], p[1]) for p in plan})} datasets).")
    for p in problems:
        print("  WARNING:", p)

    rows = []
    geo = {pl: {"inst_row": [], "inst_label": [], "inst_cov": [], "inst_box": [], "inst_boxgrid": [],
                "inst_area": [], "row_valid": [], "row_geom": []} for pl in ec.PIPELINES}
    for i, (root_name, ds, split, zs, img, seg_path) in enumerate(plan):
        seg = ec.load_slice(img, seg_path)[1]
        H, W = seg.shape
        labels = ec.slice_labels(seg)
        rows.append({"row": i, "root": root_name, "dataset": ds, "split": split, "zeroshot": zs,
                     "img_path": img, "seg_path": seg_path, "H": H, "W": W, "n_labels": len(labels)})
        for pl in ec.PIPELINES:
            g = geo[pl]
            geom = ec.make_geom(pl, H, W)
            g["row_valid"].append(ec.valid_grid(geom))
            g["row_geom"].append(geom)
            for lab in labels:
                m = seg == lab
                box = ec.box_from_mask(m)
                g["inst_row"].append(i)
                g["inst_label"].append(lab)
                g["inst_cov"].append(np.round(ec.mask_to_grid(m, geom) * 255).astype(np.uint8))
                g["inst_box"].append(box)
                g["inst_boxgrid"].append(ec.box_to_grid(ec.box_to_input(box, geom)))
                g["inst_area"].append(int(m.sum()))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(plan)} slices ({time.time() - t0:.0f} s)", flush=True)

    with open(os.path.join(args.work_dir, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for pl in ec.PIPELINES:
        g = geo[pl]
        np.savez_compressed(
            os.path.join(args.work_dir, f"geometry_{pl}.npz"),
            inst_row=np.array(g["inst_row"], np.int32), inst_label=np.array(g["inst_label"], np.float32),
            inst_cov=np.stack(g["inst_cov"]), inst_box=np.stack(g["inst_box"]).astype(np.float32),
            inst_boxgrid=np.stack(g["inst_boxgrid"]).astype(np.int16), inst_area=np.array(g["inst_area"], np.int64),
            row_valid=np.stack(g["row_valid"]), row_geom=np.array(g["row_geom"], np.int32))

    n_test = sum(r["split"] == "test" for r in rows)
    nonsq = np.mean([r["H"] != r["W"] for r in rows])
    print(f"\n{len(rows)} slices, {len(geo['sam']['inst_row'])} instances; {nonsq:.0%} non-square slices.")
    print("Disk needed per encoder condition (fp16): "
          f"neck {len(rows) * 2.0 / 1024:.1f} GB; CKA layer tokens (ViT-B, 128 tokens) "
          f"{n_test * 13 * 128 * 768 * 2 / 1024 ** 3:.1f} GB.")  # 13 = 'embed' + 12 blocks
    print(f"Saved to {args.work_dir}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
