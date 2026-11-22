#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a comparison-model result pickle (evaluation/test_comparison_models.py, e.g.
sam_med2d_fixed.pkl) into a CSV shaped like all_results.csv, so it can be analysed beside
SAM ViT-B, MedSAM and SAMRI in result_visualize_and_evaluate.ipynb. Needs pandas and numpy only.

Input pickle
  {"trained": {dataset: [ {img_name, mask_name, labels, dice, hd, msd, pixel_count,
                           area_percentage}, ... ]}, "zero_shot": {...}}
Output CSV: one row per structure
  Model, Dataset, img_path, mask_path, labels, b_dice, b_hd, b_msd, area_percentage   (= all_results.csv)
  + pixel_count, split ("trained" / "zero_shot")
  - the metrics go into the b_* columns because every comparison model here is box-prompted
    (SAMed is prompt-free and Medical-SAM-Adapter uses a point: name the model accordingly);
  - labels are projected to structure names with LABEL_PROJECTION from evaluation/utils.py, the
    same table the notebook uses; a label with no entry is written as a float ("1.0"), which is
    how the SAM / SAMRI records write it.

Optional: --merge-with all_results.csv --merged-out <new file> writes a combined table with
exactly all_results.csv's columns. Rows already there under the same --model-name are replaced.
The input all_results.csv is never modified.

Usage:
  python evaluation/comparison_pkl_to_csv.py --pkl sam_med2d_fixed/sam_med2d_fixed.pkl --model-name SAM-Med2D
  python evaluation/comparison_pkl_to_csv.py --pkl sam_med2d_fixed/sam_med2d_fixed.pkl --model-name SAM-Med2D \\
      --merge-with base/all_results.csv --merged-out base/all_results_with_sam_med2d.csv
"""

import argparse
import ast
import os
import pickle

import numpy as np
import pandas as pd

ALL_RESULTS_COLS = ["Model", "Dataset", "img_path", "mask_path", "labels", "b_dice", "b_hd", "b_msd", "area_percentage"]


def label_projection():
    """LABEL_PROJECTION from evaluation/utils.py, evaluated on its own (no plotting imports needed)."""
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils.py")).read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "LABEL_PROJECTION" for t in node.targets):
            return eval(compile(ast.Expression(node.value), "evaluation/utils.py", "eval"), {})
    return {}


def project(ds, label, proj):
    """Same rule as evaluation/utils.project_label: integer-valued label -> name, else unchanged."""
    try:
        f = float(label)
    except (TypeError, ValueError):
        return label
    key = int(f) if f.is_integer() else f
    return proj.get(ds, {}).get(key, f)


def to_frame(record, model_name, proj):
    rows, unprojected = [], set()
    for split, datasets in record.items():
        for ds, items in datasets.items():
            for it in items:
                n = len(it["labels"])
                for key in ("dice", "hd", "msd", "pixel_count", "area_percentage"):
                    if len(it[key]) != n:
                        raise ValueError(f"{ds}: {it['img_name']} has {n} labels but {len(it[key])} values of {key}")
                for j in range(n):
                    lab = project(ds, it["labels"][j], proj)
                    if not isinstance(lab, str):
                        unprojected.add(ds)
                    rows.append({"Model": model_name, "Dataset": ds, "img_path": str(it["img_name"]),
                                 "mask_path": str(it["mask_name"]), "labels": lab,
                                 "b_dice": float(it["dice"][j]), "b_hd": float(it["hd"][j]), "b_msd": float(it["msd"][j]),
                                 "area_percentage": float(it["area_percentage"][j]),
                                 "pixel_count": int(it["pixel_count"][j]), "split": split})
    return pd.DataFrame(rows), sorted(unprojected)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", required=True, help="result pickle of test_comparison_models.py")
    ap.add_argument("--model-name", required=True, help="value of the Model column, e.g. SAM-Med2D")
    ap.add_argument("--out", default=None, help="output CSV (default: the pickle's path with .csv)")
    ap.add_argument("--merge-with", default=None, help="an existing all_results.csv to combine with")
    ap.add_argument("--merged-out", default=None, help="combined CSV to write (required with --merge-with)")
    args = ap.parse_args()
    if args.merge_with and not args.merged_out:
        ap.error("--merge-with needs --merged-out (a NEW file; the input is never modified)")
    if args.merge_with and os.path.abspath(args.merge_with) == os.path.abspath(args.merged_out):
        ap.error("--merged-out must be a different file from --merge-with")

    with open(args.pkl, "rb") as f:
        record = pickle.load(f)
    if not isinstance(record, dict) or not all(isinstance(v, dict) for v in record.values()):
        raise SystemExit(f"{args.pkl} is not a comparison result pickle ({{'trained': {{dataset: [...]}}, ...}})")
    df, unprojected = to_frame(record, args.model_name, label_projection())
    if df.empty:
        raise SystemExit(f"{args.pkl} holds no structures")

    out = args.out or os.path.splitext(args.pkl)[0] + ".csv"
    df[ALL_RESULTS_COLS + ["pixel_count", "split"]].to_csv(out, index=False)
    print(f"{args.model_name}: {len(df)} structures from {df['img_path'].nunique()} slices -> {out}")
    for split, g in df.groupby("split", sort=False):
        med = g.groupby("Dataset")["b_dice"].median()
        print(f"  [{split}] {len(med)} datasets, {len(g)} structures | median Dice over structures {g['b_dice'].median():.3f} | "
              f"mean of dataset medians {med.mean():.3f}")
        for ds, v in med.items():
            print(f"    {ds:32s} {v:.3f}  (n={int((g['Dataset'] == ds).sum())})")
    if unprojected:
        print("  labels kept as numbers (no entry in LABEL_PROJECTION):", ", ".join(unprojected))
    bad = int((~np.isfinite(df[["b_hd", "b_msd"]])).any(axis=1).sum())
    if bad:
        print(f"  note: {bad} structures have a non-finite HD or MSD (empty prediction); Dice is unaffected")

    if args.merge_with:
        base = pd.read_csv(args.merge_with)
        missing = [c for c in ALL_RESULTS_COLS if c not in base.columns]
        if missing:
            raise SystemExit(f"{args.merge_with} lacks the columns {missing} - not an all_results.csv")
        replaced = int((base["Model"] == args.model_name).sum())
        base = base[base["Model"] != args.model_name]
        merged = pd.concat([base[ALL_RESULTS_COLS], df[ALL_RESULTS_COLS]], ignore_index=True)
        merged.to_csv(args.merged_out, index=False)
        print(f"merged with {args.merge_with}: {len(base)} existing rows + {len(df)} new"
              + (f" ({replaced} old {args.model_name} rows replaced)" if replaced else "")
              + f" -> {args.merged_out}")
        print("  rows per model:", merged["Model"].value_counts().to_dict())


if __name__ == "__main__":
    main()
