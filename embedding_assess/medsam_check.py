#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPU-only check of how MedSAM's MR intensity preprocessing changes the 2D test
slices, compared with plain min-max normalisation.

For every test slice (and every label in its mask) three versions of the image
are built, all in [0, 255]:

  minmax      Plain per-slice min-max to [0, 255] (what SAM / SAMRI receive,
              utils/dataloader.NiiDataset._preprocess).
  percentile  MedSAM's MR preprocessing (MedSAM-main/pre_CT_MR.py, non-CT
              branch): clip to the 0.5-99.5 percentile of non-zero pixels,
              rescale to [0, 255], keep zero background at 0. Identical to
              medsam_mr_preprocess() in evaluation/test_medsam.py.
  old         What the previous evaluation/test_medsam.py fed MedSAM: raw
              intensities cast to uint8 BEFORE min-max normalisation. Values
              above 255 wrap around. By default this is emulated at native
              resolution; --exact-old-pipeline reproduces the original 1024x1024
              bicubic resize + cast (slower, needs scikit-image).

Outputs (in --save-path):
  instances.csv        one row per (slice, label): slice-level intensity stats,
                       differences between the versions, and the target-vs-
                       surround contrast inside the tight GT box under each
                       version (what a box-prompted model has to separate).
  dataset_summary.csv  one row per dataset (medians / fractions).
  overall_summary.txt  headline numbers over all instances (printed too).
  examples/            optional PNG panels per dataset (needs matplotlib).

Usage on a CPU node (run from the repo root or anywhere). Pass one or more test
roots, each holding <dataset>/<split>/ folders; datasets are reported as
<root folder name>/<dataset>, e.g. SAMRI_train_test/ACDC, Zeroshot/<dataset>.
  # quick check first (20 slices per dataset, a few minutes at most):
  python embedding_assess/medsam_check.py \
      --test-image-path /scratch/user/s4670484/Datasets/SAMRI_train_test/ \
                        /scratch/user/s4670484/Datasets/Zeroshot/ \
      --save-path /scratch/user/s4670484/Eval_results/medsam_check_quick/ \
      --max-slices-per-dataset 20
  # full run:
  python embedding_assess/medsam_check.py \
      --test-image-path /scratch/user/s4670484/Datasets/SAMRI_train_test/ \
                        /scratch/user/s4670484/Datasets/Zeroshot/ \
      --save-path /scratch/user/s4670484/Eval_results/medsam_check_results/
"""

import argparse
import csv
import os
import random
import sys
import time
from glob import glob
from multiprocessing import Pool

import numpy as np
import nibabel as nib

IMAGE_KEYS = "*_img_*"   # same keys as utils/utils.py
MASK_KEYS = "*_seg_*"
EPS = 1e-8


# --------------------------------------------------------------------------
# The three image versions
# --------------------------------------------------------------------------
def minmax_255(img):
    """Plain min-max to [0, 255] (SAM / SAMRI input, before uint8 rounding)."""
    img = img.astype(np.float64)
    return (img - img.min()) / (img.max() - img.min() + EPS) * 255.0


def percentile_255(img):
    """MedSAM MR preprocessing; must match medsam_mr_preprocess() in
    evaluation/test_medsam.py (kept as float here, before the uint8 cast)."""
    image_data = img.astype(np.float64)
    foreground = image_data[image_data > 0]
    if foreground.size == 0:
        return np.zeros_like(image_data)
    lower_bound = np.percentile(foreground, 0.5)
    upper_bound = np.percentile(foreground, 99.5)
    image_data_pre = np.clip(image_data, lower_bound, upper_bound)
    image_data_pre = (
        (image_data_pre - np.min(image_data_pre))
        / max(np.max(image_data_pre) - np.min(image_data_pre), EPS)
        * 255.0
    )
    image_data_pre[image_data == 0] = 0
    return image_data_pre


def old_pipeline_255(img, exact=False):
    """Previous test_medsam.py behaviour: uint8 cast before normalisation.
    Returns the image at native resolution in [0, 255]."""
    img = img.astype(np.float64)
    if exact:
        from skimage import transform  # only needed for the exact mode
        H, W = img.shape
        img_3c = np.repeat(img[:, :, None], 3, axis=-1)
        r = transform.resize(img_3c, (1024, 1024), order=3,
                             preserve_range=True, anti_aliasing=True).astype(np.uint8)
        r = (r - r.min()) / np.clip(r.max() - r.min(), a_min=EPS, a_max=None) * 255.0
        # back to native size so it can be compared pixel-wise with the others
        return transform.resize(r[..., 0], (H, W), order=1, preserve_range=True,
                                anti_aliasing=True)
    u = img.astype(np.uint8)  # same cast as the old script (platform behaviour)
    u = u.astype(np.float64)
    return (u - u.min()) / max(u.max() - u.min(), EPS) * 255.0


# --------------------------------------------------------------------------
# Per-slice work
# --------------------------------------------------------------------------
def _safe_corr(a, b):
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _contrast(img, target, surround):
    """|mean(target) - mean(surround)| in 0-255 units."""
    if target.sum() == 0 or surround.sum() == 0:
        return float("nan")
    return float(abs(img[target].mean() - img[surround].mean()))


def process_slice(job):
    ds_name, img_path, seg_path, exact_old = job
    try:
        raw = np.asarray(nib.load(img_path).get_fdata())[0]
        seg = np.asarray(nib.load(seg_path).get_fdata())[0]
    except Exception as e:  # report, never silently drop
        return [], f"{img_path}: {e}"

    if raw.shape != seg.shape:
        return [], f"{img_path}: image {raw.shape} vs mask {seg.shape}"

    H, W = raw.shape
    fg = raw > 0
    v_mm = minmax_255(raw)
    v_pc = percentile_255(raw)
    v_old = old_pipeline_255(raw, exact=exact_old)

    nz = raw[fg]
    p_lo = float(np.percentile(nz, 0.5)) if nz.size else float("nan")
    p_hi = float(np.percentile(nz, 99.5)) if nz.size else float("nan")
    on = fg if fg.any() else np.ones_like(fg)  # pixels the comparison is made on

    slice_stats = {
        "dataset": ds_name,
        "img_path": img_path,
        "H": H, "W": W, "is_square": int(H == W),
        "raw_min": float(raw.min()), "raw_max": float(raw.max()),
        "raw_p0.5": p_lo, "raw_p99.5": p_hi,
        "max_over_p99.5": float(raw.max() / p_hi) if p_hi and p_hi > 0 else float("nan"),
        "is_float_0_1": int(raw.min() >= 0 and raw.max() <= 1.0),
        "has_negative": int(raw.min() < 0),
        "frac_raw_gt_255": float((raw > 255).mean()),
        "frac_clipped_by_percentile": float(((raw > p_hi) | ((raw < p_lo) & fg)).mean()) if nz.size else float("nan"),
        # minmax vs percentile (the question asked)
        "mae_minmax_vs_pct": float(np.abs(v_mm - v_pc)[on].mean()),
        "p99_tissue_minmax": float(np.percentile(v_mm[on], 99)),
        "p99_tissue_pct": float(np.percentile(v_pc[on], 99)),
        # old pipeline vs the correct versions
        "mae_old_vs_pct": float(np.abs(v_old - v_pc)[on].mean()),
        "corr_old_vs_minmax": _safe_corr(v_old[on].ravel(), v_mm[on].ravel()),
    }

    rows = []
    for label in np.unique(seg):
        if label == 0:
            continue
        m = seg == label
        ys, xs = np.nonzero(m)
        box = np.zeros_like(m)
        box[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True  # tight box, as in testing
        surround = box & ~m
        c_mm = _contrast(v_mm, m, surround)
        c_pc = _contrast(v_pc, m, surround)
        c_old = _contrast(v_old, m, surround)
        row = dict(slice_stats)
        row.update({
            "label": float(label),
            "mask_pixels": int(m.sum()),
            "box_fill": float(m.sum() / box.sum()),
            "contrast_minmax": c_mm,
            "contrast_pct": c_pc,
            "contrast_old": c_old,
            "contrast_ratio_pct_over_minmax": c_pc / c_mm if c_mm and c_mm > 0 else float("nan"),
            "mae_minmax_vs_pct_in_box": float(np.abs(v_mm - v_pc)[box].mean()),
        })
        rows.append(row)
    return rows, None


# --------------------------------------------------------------------------
# Example panels
# --------------------------------------------------------------------------
def save_example(ds_name, img_path, seg_path, out_dir, exact_old):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    raw = np.asarray(nib.load(img_path).get_fdata())[0]
    seg = np.asarray(nib.load(seg_path).get_fdata())[0]
    v_mm, v_pc, v_old = minmax_255(raw), percentile_255(raw), old_pipeline_255(raw, exact_old)
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.4))
    for a, im, t in zip(ax[:3], [v_mm, v_pc, v_old],
                        ["min-max (SAM/SAMRI)", "percentile 0.5-99.5 (MedSAM)",
                         "old test_medsam.py (cast first)"]):
        a.imshow(im, cmap="gray", vmin=0, vmax=255)
        a.contour(seg > 0, levels=[0.5], colors="r", linewidths=0.6)
        a.set_title(t, fontsize=10)
        a.axis("off")
    d = ax[3].imshow(np.abs(v_mm - v_pc), cmap="magma", vmin=0, vmax=255)
    ax[3].set_title("|min-max - percentile|", fontsize=10)
    ax[3].axis("off")
    fig.colorbar(d, ax=ax[3], fraction=0.046)
    fig.suptitle(f"{ds_name}: {os.path.basename(img_path)}  raw [{raw.min():.1f}, {raw.max():.1f}]",
                 fontsize=10)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, os.path.basename(img_path).replace(".nii.gz", ".png")), dpi=90)
    plt.close(fig)


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------
def _med(vals):
    v = np.asarray([x for x in vals if x == x], dtype=float)  # drop NaN
    return float(np.median(v)) if v.size else float("nan")


def _frac(vals, cond):
    v = [x for x in vals if x == x]
    return float(np.mean([cond(x) for x in v])) if v else float("nan")


def summarise(rows):
    """Slice-level stats are taken once per slice; contrast stats per instance."""
    slices = {r["img_path"]: r for r in rows}.values()
    col = lambda rs, k: [r[k] for r in rs]
    return {
        "n_slices": len(slices),
        "n_instances": len(rows),
        "frac_slices_raw_max_gt_255": _frac(col(slices, "raw_max"), lambda x: x > 255),
        "frac_slices_float_0_1": _frac(col(slices, "is_float_0_1"), lambda x: x == 1),
        "frac_slices_negative": _frac(col(slices, "has_negative"), lambda x: x == 1),
        "frac_slices_non_square": _frac(col(slices, "is_square"), lambda x: x == 0),
        "median_raw_max": _med(col(slices, "raw_max")),
        "median_max_over_p99.5": _med(col(slices, "max_over_p99.5")),
        "median_mae_minmax_vs_pct": _med(col(slices, "mae_minmax_vs_pct")),
        "median_p99_tissue_minmax": _med(col(slices, "p99_tissue_minmax")),
        "median_p99_tissue_pct": _med(col(slices, "p99_tissue_pct")),
        "median_contrast_minmax": _med(col(rows, "contrast_minmax")),
        "median_contrast_pct": _med(col(rows, "contrast_pct")),
        "median_contrast_ratio_pct_over_minmax": _med(col(rows, "contrast_ratio_pct_over_minmax")),
        "frac_instances_contrast_higher_with_pct": _frac(col(rows, "contrast_ratio_pct_over_minmax"), lambda x: x > 1),
        "median_mae_old_vs_pct": _med(col(slices, "mae_old_vs_pct")),
        "frac_slices_old_corrupted_corr_lt_0.9": _frac(col(slices, "corr_old_vs_minmax"), lambda x: x < 0.9),
        "median_contrast_old": _med(col(rows, "contrast_old")),
    }


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def collect_jobs(roots, split, max_per_ds, seed, exact_old):
    rng = random.Random(seed)
    jobs, per_ds, problems = [], {}, []
    for root in roots:
        root_name = os.path.basename(root.rstrip("/"))
        if not os.path.isdir(root):
            problems.append(f"{root}: not a directory")
            continue
        for ds in sorted(glob(os.path.join(root, "*"))):
            folder = os.path.join(ds, split)
            if not os.path.isdir(folder):
                continue
            name = f"{root_name}/{os.path.basename(ds.rstrip('/'))}"
            imgs = sorted(glob(os.path.join(folder, IMAGE_KEYS)))
            segs = sorted(glob(os.path.join(folder, MASK_KEYS)))
            if len(imgs) != len(segs):
                problems.append(f"{name}: {len(imgs)} images vs {len(segs)} masks")
            pairs = list(zip(imgs, segs))
            # same pairing as NiiDataset (sorted lists); check names actually match
            bad = [p for p in pairs
                   if os.path.basename(p[0]).replace("_img_", "_seg_") != os.path.basename(p[1])]
            if bad:
                problems.append(f"{name}: {len(bad)} image/mask name mismatches, e.g. {bad[0]}")
            if max_per_ds and len(pairs) > max_per_ds:
                pairs = rng.sample(pairs, max_per_ds)
            per_ds[name] = pairs
            jobs += [(name, i, s, exact_old) for i, s in pairs]
    return jobs, per_ds, problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-image-path", required=True, nargs="+",
                    help="One or more roots holding <dataset>/<split>/ folders, "
                         "e.g. .../SAMRI_train_test/ .../Zeroshot/")
    ap.add_argument("--split", default="testing", choices=["training", "validation", "testing"])
    ap.add_argument("--save-path", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                        "medsam_check_results"))
    ap.add_argument("--max-slices-per-dataset", type=int, default=0,
                    help="Random sample per dataset (0 = all slices). Use ~20 for a quick check.")
    ap.add_argument("--n-examples", type=int, default=3, help="PNG example panels per dataset (0 = none).")
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)),
                    help="CPU processes (default: $SLURM_CPUS_PER_TASK, else 4).")
    ap.add_argument("--exact-old-pipeline", action="store_true",
                    help="Reproduce the old 1024x1024 resize + cast exactly (slow, needs scikit-image).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.save_path, exist_ok=True)
    jobs, per_ds, problems = collect_jobs(args.test_image_path, args.split,
                                          args.max_slices_per_dataset, args.seed,
                                          args.exact_old_pipeline)
    print(f"Found {len(per_ds)} datasets, {len(jobs)} slices to check "
          f"({'all' if not args.max_slices_per_dataset else f'up to {args.max_slices_per_dataset}'} per dataset), "
          f"{args.workers} workers.")
    for p in problems:
        print("  WARNING:", p)
    if not jobs:
        sys.exit("No slices found: check --test-image-path (each root should contain <dataset>/" + args.split + "/).")

    rows, errors = [], []
    with Pool(args.workers) as pool:
        for i, (r, err) in enumerate(pool.imap_unordered(process_slice, jobs, chunksize=16), 1):
            rows += r
            if err:
                errors.append(err)
            if i % 2000 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} slices done ({time.time() - t0:.0f} s)", flush=True)

    rows.sort(key=lambda r: (r["dataset"], r["img_path"], r["label"]))
    write_csv(os.path.join(args.save_path, "instances.csv"), rows)

    ds_rows = []
    for name in sorted(per_ds):
        rs = [r for r in rows if r["dataset"] == name]
        if rs:
            ds_rows.append({"dataset": name, **summarise(rs)})
    write_csv(os.path.join(args.save_path, "dataset_summary.csv"), ds_rows)

    overall = summarise(rows)
    lines = ["OVERALL (all datasets)"] + [f"  {k}: {v:.4g}" if isinstance(v, float) else f"  {k}: {v}"
                                         for k, v in overall.items()]
    lines += ["", "PER DATASET (slices | raw max median | max/p99.5 | MAE min-max vs pct | "
                  "contrast ratio pct/min-max | old-pipeline corrupted (corr<0.9))"]
    for d in ds_rows:
        lines.append(f"  {d['dataset']:32s} {d['n_slices']:6d} | {d['median_raw_max']:9.1f} | "
                     f"{d['median_max_over_p99.5']:5.2f} | {d['median_mae_minmax_vs_pct']:5.1f} | "
                     f"{d['median_contrast_ratio_pct_over_minmax']:5.2f} | "
                     f"{d['frac_slices_old_corrupted_corr_lt_0.9']:5.1%}")
    if errors:
        lines += ["", f"{len(errors)} slices could not be read (first 10):"] + ["  " + e for e in errors[:10]]
    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(args.save_path, "overall_summary.txt"), "w") as f:
        f.write(report + "\n")

    if args.n_examples:
        rng = random.Random(args.seed)
        for name, pairs in per_ds.items():
            for img_path, seg_path in rng.sample(pairs, min(args.n_examples, len(pairs))):
                save_example(name, img_path, seg_path,
                             os.path.join(args.save_path, "examples", name), args.exact_old_pipeline)

    print(f"\nSaved to {args.save_path}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
