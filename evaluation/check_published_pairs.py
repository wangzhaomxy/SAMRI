#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Were the published per-structure results computed on correctly paired image/mask slices?
(CPU, seconds; no model.)

Why: the test pipeline (utils/dataloader.NiiDataset, and the NiiDataset copy in
evaluation/test_medsam.py) pairs the k-th sorted image with the k-th sorted mask,
NOT by name. One surplus or missing mask in a testing/ folder therefore shifts every
later image onto the next slice's mask without any error (a surplus image crashes
instead, since the length is the image count). On thin structures such as shoulder
cartilage that collapses the Dice. Published SAM ViT-B on MSK_shoulder is 0.267 while
the rerun gives 0.656 on the same structures; every other dataset agrees.

Checks
  1. published CSV (all_results.csv or a per-model CSV from result_visualize_and_evaluate.ipynb):
     per model x dataset, how many rows have an image and mask from DIFFERENT slices
     (name with _img_ -> _seg_ differs), the typical slice offset, and median Dice of
     matched vs mismatched rows
  2. with --rerun-pickle (test_vis.py output, e.g. sam_msk_rerun/sam_vit_b_01ec64): the
     same check on the rerun, and published vs rerun Dice per structure (joined on mask
     file + label)
  3. with --dataset-root: unpaired files in each <dataset>/testing/ folder NOW

Usage (Bunya, any env with pandas):
  python evaluation/check_published_pairs.py \
      --published-csv <results folder>/base/all_results.csv \
      --rerun-pickle /scratch/user/s4670484/Eval_results/sam_msk_rerun/sam_vit_b_01ec64 \
      --dataset-root /scratch/user/s4670484/Datasets/Zeroshot/ \
      --datasets MSK_shoulder CHAOS_T1
"""

import argparse
import ast
import os
import pickle
import re
from glob import glob

import numpy as np
import pandas as pd

SLICE = re.compile(r"_(?:img|seg)_(\d+)\.nii\.gz$")


def label_projection():
    """LABEL_PROJECTION from evaluation/utils.py, evaluated on its own so the module's plotting
    imports are not needed (it contains dict comprehensions, so literal_eval cannot read it)."""
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils.py")).read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "LABEL_PROJECTION" for t in node.targets):
            return eval(compile(ast.Expression(node.value), "evaluation/utils.py", "eval"), {})
    return {}


def slice_no(path):
    m = SLICE.search(os.path.basename(str(path)))
    return int(m.group(1)) if m else None


def add_pair_columns(df):
    img = df["img_path"].astype(str).map(os.path.basename)
    msk = df["mask_path"].astype(str).map(os.path.basename)
    df = df.copy()
    df["paired"] = img.str.replace("_img_", "_seg_", regex=False) == msk
    si, sm = df["img_path"].map(slice_no), df["mask_path"].map(slice_no)
    df["offset"] = [(b - a) if a is not None and b is not None else np.nan for a, b in zip(si, sm)]
    return df


def report_pairs(df, title, dice_col="b_dice"):
    print(f"\n{title}")
    print(f"  {'model':12s} {'dataset':28s} {'rows':>6s} {'mismatched':>11s}  {'offsets (mask - image slice)':30s} "
          f"{'median Dice matched | mismatched':>33s}")
    for (model, ds), g in df.groupby(["Model", "Dataset"], sort=True):
        bad = g[~g["paired"]]
        offs = bad["offset"].dropna().astype(int).value_counts().head(3)
        off_txt = ", ".join(f"{k:+d} x{v}" for k, v in offs.items()) if len(offs) else "-"
        med_ok = g.loc[g["paired"], dice_col].median()
        med_bad = bad[dice_col].median() if len(bad) else float("nan")
        flag = "  <-- SHIFTED PAIRS" if len(bad) else ""
        print(f"  {str(model):12s} {str(ds):28s} {len(g):6d} {len(bad):5d} ({len(bad) / len(g):4.0%})  {off_txt:30s} "
              f"{med_ok:15.3f} | {med_bad:.3f}{flag}")
        if len(bad):
            r = bad.iloc[0]
            print(f"      e.g. image {os.path.basename(str(r['img_path']))}  with mask {os.path.basename(str(r['mask_path']))}")


def rerun_frame(path, proj):
    rec = pickle.load(open(path, "rb"))
    rows = []
    for ds, items in rec.items():
        for it in items:
            for lab, d in zip(it["labels"], it["b_dice"]):
                key = int(lab) if float(lab).is_integer() else lab
                rows.append({"Model": "rerun", "Dataset": ds, "img_path": it["img_name"], "mask_path": it["mask_name"],
                             "labels": proj.get(ds, {}).get(key, key), "b_dice": float(d)})
    return pd.DataFrame(rows)


def unpaired_files(root, datasets):
    print(f"\nUnpaired files in <dataset>/testing/ under {root} NOW (name-based)")
    for ds_dir in sorted(d for d in glob(os.path.join(root, "*")) if os.path.isdir(d)):
        ds = os.path.basename(ds_dir)
        if datasets and ds not in datasets:
            continue
        t = os.path.join(ds_dir, "testing")
        imgs = {os.path.basename(p) for p in glob(os.path.join(t, "*_img_*"))}
        segs = {os.path.basename(p) for p in glob(os.path.join(t, "*_seg_*"))}
        no_mask = sorted(i for i in imgs if i.replace("_img_", "_seg_") not in segs)
        no_img = sorted(s for s in segs if s.replace("_seg_", "_img_") not in imgs)
        state = "ok" if not (no_mask or no_img) else "UNPAIRED - sorted-order pairing would shift"
        print(f"  {ds:28s} {len(imgs):6d} images {len(segs):6d} masks | images without mask {len(no_mask)}, "
              f"masks without image {len(no_img)} -> {state}")
        for name in (no_mask + no_img)[:3]:
            print(f"      {name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--published-csv", required=True)
    ap.add_argument("--rerun-pickle", default=None)
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--datasets", nargs="*", default=None, help="limit to these datasets (default: all)")
    ap.add_argument("--models", nargs="*", default=None, help="limit to these Model values (default: all)")
    args = ap.parse_args()

    pub = pd.read_csv(args.published_csv)
    if args.datasets:
        pub = pub[pub["Dataset"].isin(args.datasets)]
    if args.models:
        pub = pub[pub["Model"].isin(args.models)]
    pub = add_pair_columns(pub)
    report_pairs(pub, f"PUBLISHED  {args.published_csv}")

    if args.rerun_pickle:
        rr = rerun_frame(args.rerun_pickle, label_projection())
        if args.datasets:
            rr = rr[rr["Dataset"].isin(args.datasets)]
        rr = add_pair_columns(rr)
        report_pairs(rr, f"RERUN  {args.rerun_pickle}")
        print("\nPublished vs rerun, per structure (joined on mask file name + label)")
        for model, g in pub.groupby("Model"):
            a = g.assign(mask=g["mask_path"].astype(str).map(os.path.basename), lab=g["labels"].astype(str))
            b = rr.assign(mask=rr["mask_path"].astype(str).map(os.path.basename), lab=rr["labels"].astype(str))
            j = a.merge(b, on=["Dataset", "mask", "lab"], suffixes=("_pub", "_rerun"))
            for ds, h in j.groupby("Dataset"):
                same_img = (h["img_path_pub"].astype(str).map(os.path.basename)
                            == h["img_path_rerun"].astype(str).map(os.path.basename))
                with np.errstate(invalid="ignore", divide="ignore"):  # constant columns -> nan, not a warning
                    corr = np.corrcoef(h["b_dice_pub"], h["b_dice_rerun"])[0, 1] if len(h) > 2 else float("nan")
                print(f"  {str(model):12s} {ds:28s} {len(h):5d} structures | median Dice published "
                      f"{h['b_dice_pub'].median():.3f}, rerun {h['b_dice_rerun'].median():.3f} | correlation {corr:.3f} | "
                      f"same image file {same_img.mean():.0%}")

    if args.dataset_root:
        unpaired_files(args.dataset_root, args.datasets)


if __name__ == "__main__":
    main()
