#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data size at a FIXED training budget (CPU, seconds): does more data per dataset help when every data size
trains for the same number of training samples?

Why: in the original data-size runs (runs_curve, arms sam-enc_sam-dec_n<N>) each run stops early on the shared
validation slices, so more data also means longer training (best epoch after ~1-4k training samples with 15
slices per dataset, ~0.2-0.8 M with 5,000). The fixed-budget runs (train_decoder_curve.py --fixed-samples, arms
sam-enc_sam-dec_n<N>_s<B>) separate the two: size N and budget B are crossed, no early stopping, no best-epoch
choice, the same slices as the original runs for each seed.

Inputs
  --instances         eval_decoder.py's eval_instances.csv for runs_fixed (in-domain, zero-shot, knee test sets)
  --runs              runs_fixed (meta.json per run: samples seen, epochs, validation Dice, training time)
  --reference         optional: eval_instances.csv holding the original early-stopped arms (results_knee)
  --reference-runs    optional: runs_curve (their best epoch -> training samples to the chosen decoder)
Metric: mean over datasets of the per-dataset median Dice, per seed (as eval_summary.txt); one structure
  (e.g. scapula cartilage) = median over its test instances, per seed; then mean +- sd over the 3 seeds.
  Columns: in-domain (the 11 datasets), zero-shot* (the zero-shot sets without MSK_shoulder), unseen knee
  (MSK_PD + MSK_FLASH), FLASH femoral cartilage, scapula and humerus cartilage (MSK_shoulder).
Comparisons: largest vs smallest data size at each budget, seed-paired (seed k of both sizes; seed k's small
  set is part of its large set); datasets = Wilcoxon over the per-dataset median differences of seed-averaged
  structures, as in eval_decoder.py; "meaningful" = |difference| > 0.01 and > the seed spread. Slopes = least
  squares of the per-seed score on log10(size) at each budget (gain per 10x more slices per dataset), and on
  log10(budget) at each size (gain per 10x longer training); standard errors treat the seeds as independent.
Outputs in --out-dir: summary.txt (headline first), fixed_grid.csv, fixed_validation.csv.
"""

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from glob import glob

import numpy as np

MEANINGFUL = 0.01
ARM = re.compile(r"sam-enc_sam-dec(-lora)?_n(\d+)_s(\d+)")
REF = re.compile(r"sam-enc_sam-dec(-lora)?_n(\d+)")
# (column, test set, datasets kept (None = all), datasets dropped, label (None = all structures))
COLS = [("in-domain", "in-domain", None, (), None),
        ("zero-shot*", "zero-shot", None, ("MSK_shoulder",), None),
        ("unseen knee", "knee", ("MSK_PD", "MSK_FLASH"), (), None),
        ("FLASH femoral cart.", "knee", ("MSK_FLASH",), (), 2.0),
        ("scapula cart.", "zero-shot", ("MSK_shoulder",), (), 180.0),
        ("humerus cart.", "zero-shot", ("MSK_shoulder",), (), 225.0)]


def load(paths, keep):
    by = defaultdict(list)  # (system, seed, test set, dataset, label) -> dice
    names = {}
    for p in paths:
        with open(p) as f:
            for r in csv.DictReader(f):
                if not keep(r["system"]):
                    continue
                names[r["system"]] = r["name"]
                by[(r["system"], int(r["seed"]), r["test_set"], r["dataset"], float(r["label"]))].append(
                    (int(r["row"]), float(r["dice"])))
    return by, names


def ms(v, d=3):
    v = [x for x in v if x == x]
    if not v:
        return "-"
    return f"{np.mean(v):.{d}f}" + (f"±{np.std(v, ddof=1):.{d}f}" if len(v) > 1 else "")


def kfmt(x):
    """6000 -> 6k, 1234 -> 1.2k, 600000 -> 600k, 1.5e6 -> 1.50M"""
    if x != x:
        return "-"
    if x < 1000:
        return f"{x:.0f}"
    if x < 10000:
        return f"{x / 1000:.1f}k"
    if x < 1e6:
        return f"{x / 1000:.0f}k"
    return f"{x / 1e6:.2f}M"


def slope(x, y):
    """least-squares slope of y on x and its standard error"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~np.isnan(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.ptp(x) == 0:
        return float("nan"), float("nan")
    X = np.c_[np.ones_like(x), x]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ coef
    s2 = res @ res / (len(x) - 2)
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    return float(coef[1]), float(se)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances", nargs="+", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--reference", nargs="*", default=[])
    ap.add_argument("--reference-runs", default=None)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    from scipy.stats import wilcoxon

    by, names = load(args.instances, lambda s: ARM.fullmatch(s) is not None)
    if not names:
        raise SystemExit(f"no fixed-budget arms (sam-enc_sam-dec_n<N>_s<B>) in {args.instances}")
    ref_by, ref_names = load(args.reference, lambda s: REF.fullmatch(s) is not None) if args.reference else ({}, {})
    metas = {}
    for p in glob(os.path.join(args.runs, "*", "seed*", "meta.json")):
        m = json.load(open(p))
        metas[(m["arm"], int(m["seed"]))] = m
    ref_metas = {}
    if args.reference_runs:
        for p in glob(os.path.join(args.reference_runs, "*", "seed*", "meta.json")):
            m = json.load(open(p))
            if REF.fullmatch(m["arm"]):
                ref_metas[(m["arm"], int(m["seed"]))] = m

    # arms: (lora, n, budget) -> system id
    grid = {}
    for s in names:
        g = ARM.fullmatch(s)
        grid[(bool(g.group(1)), int(g.group(2)), int(g.group(3)))] = s
    modes = sorted({k[0] for k in grid})
    sizes = sorted({k[1] for k in grid})
    budgets = sorted({k[2] for k in grid})

    def seeds(store, system):
        return sorted({k[1] for k in store if k[0] == system})

    def score(store, system, seed, col):
        _, t, keep, drop, label = col
        per_ds = defaultdict(list)
        for (s, sd, tt, ds, lab), v in store.items():
            if s != system or sd != seed or tt != t or ds in drop or (keep is not None and ds not in keep):
                continue
            if label is not None and lab != label:
                continue
            per_ds[ds] += [d for _, d in v]
        if not per_ds:
            return float("nan")
        return float(np.mean([np.nanmedian(v) for v in per_ds.values()]))

    memo = {}

    def per_seed(store, system, col):
        key = (id(store), system, col[0])
        if key not in memo:
            memo[key] = [score(store, system, sd, col) for sd in seeds(store, system)]
        return memo[key]

    def structures(store, system, col):
        """seed-averaged Dice per structure: (dataset, row, label) -> dice"""
        _, t, keep, drop, label = col
        acc = defaultdict(list)
        for (s, sd, tt, ds, lab), v in store.items():
            if s != system or tt != t or ds in drop or (keep is not None and ds not in keep):
                continue
            if label is not None and lab != label:
                continue
            for row, d in v:
                acc[(ds, row, lab)].append(d)
        return {k: float(np.nanmean(v)) for k, v in acc.items()}

    def compare(a, b, col):
        """a - b: seed-paired score differences; datasets: per-dataset median of structure differences"""
        sa, sb = seeds(by, a), seeds(by, b)
        common = [sd for sd in sa if sd in sb]
        diffs = [score(by, a, sd, col) - score(by, b, sd, col) for sd in common]
        xa, xb = structures(by, a, col), structures(by, b, col)
        d_ds = defaultdict(list)
        for k, v in xa.items():
            if k in xb:
                d_ds[k[0]].append(v - xb[k])
        ds_med = np.array([np.nanmedian(v) for v in d_ds.values()])
        p = wilcoxon(ds_med).pvalue if len(ds_med) > 5 and np.any(ds_med != 0) else float("nan")
        spread = max([np.ptp(per_seed(by, s, col)) if len(seeds(by, s)) > 1 else 0.0 for s in (a, b)])
        mean = float(np.mean(diffs)) if diffs else float("nan")
        verdict = "meaningful" if abs(mean) > max(MEANINGFUL, spread) else "within noise"
        return diffs, ds_med, p, spread, verdict

    lines = []
    rows_csv, val_csv = [], []
    for lora in modes:
        what = "LoRA on decoder" if lora else "decoder trained"
        lo, hi = sizes[0], sizes[-1]
        # ---- headline: largest vs smallest data size at each budget ------------------------------------------
        lines.append(f"HEADLINE ({what}): {hi:,} vs {lo:,} slices per dataset at the SAME training budget "
                     f"(seed-paired difference, mean ± sd over seeds; 'meaningful' = > {MEANINGFUL} and > seed spread)")
        for col in COLS[:3]:
            lines.append(f"  [{col[0]}]")
            for b in budgets:
                a, c = grid.get((lora, hi, b)), grid.get((lora, lo, b))
                if not a or not c:
                    continue
                diffs, ds_med, p, spread, verdict = compare(a, c, col)
                ds_part = (f" | datasets: {int((ds_med > 0).sum())}/{len(ds_med)} higher, Wilcoxon p={p:.2g}"
                           if len(ds_med) else "")
                lines.append(f"    budget {b:>9,}: {ms(diffs, 4):>16s} ({hi:,}: {ms(per_seed(by, a, col))}; "
                             f"{lo:,}: {ms(per_seed(by, c, col))}){ds_part} -> {verdict}")
        lines.append("")
        # ---- slopes ---------------------------------------------------------------------------------------------
        lines.append(f"GAIN PER 10x MORE SLICES PER DATASET at each fixed budget ({what}; slope ± SE over sizes x seeds)")
        lines.append("  " + f"{'budget':>10s} " + " ".join(f"{c[0]:>20s}" for c in COLS))
        for b in budgets:
            cells = []
            for col in COLS:
                xs, ys = [], []
                for n in sizes:
                    s = grid.get((lora, n, b))
                    if s:
                        v = per_seed(by, s, col)
                        xs += [np.log10(n)] * len(v)
                        ys += v
                k, se = slope(xs, ys)
                cells.append(f"{k:+.4f}±{se:.4f}" if k == k else "-")
            lines.append("  " + f"{b:>10,} " + " ".join(f"{c:>20s}" for c in cells))
        lines.append(f"GAIN PER 10x LONGER TRAINING at each fixed data size ({what}; slope ± SE over budgets x seeds)")
        lines.append("  " + f"{'per dataset':>11s} " + " ".join(f"{c[0]:>20s}" for c in COLS))
        for n in sizes:
            cells = []
            for col in COLS:
                xs, ys = [], []
                for b in budgets:
                    s = grid.get((lora, n, b))
                    if s:
                        v = per_seed(by, s, col)
                        xs += [np.log10(b)] * len(v)
                        ys += v
                k, se = slope(xs, ys)
                cells.append(f"{k:+.4f}±{se:.4f}" if k == k else "-")
            lines.append("  " + f"{n:>11,} " + " ".join(f"{c:>20s}" for c in cells))
        lines.append("")

        # ---- grids --------------------------------------------------------------------------------------------
        for col in COLS:
            lines.append(f"{col[0].upper()} ({what}): rows = slices per dataset, columns = training budget "
                         f"(samples); mean ± sd over seeds"
                         + ("; last column = the original run, stopped early on validation" if ref_names else ""))
            head = "  " + f"{'per dataset':>11s} " + " ".join(f"{kfmt(b):>14s}" for b in budgets)
            if ref_names:
                head += f" {'early-stopped':>14s} {'its samples*':>12s}"
            lines.append(head)
            for n in sizes:
                cells = []
                for b in budgets:
                    s = grid.get((lora, n, b))
                    cells.append(ms(per_seed(by, s, col)) if s else "-")
                line = "  " + f"{n:>11,} " + " ".join(f"{c:>14s}" for c in cells)
                if ref_names:
                    r = f"sam-enc_sam-dec{'-lora' if lora else ''}_n{n}"
                    rs = ms(per_seed(ref_by, r, col)) if r in ref_names else "-"
                    smp = [ref_metas[(r, sd)]["best_epoch"] * ref_metas[(r, sd)]["training_slices"]
                           for sd in range(10) if (r, sd) in ref_metas]
                    line += f" {rs:>14s} {(kfmt(np.mean(smp)) if smp else '-'):>12s}"
                lines.append(line)
            if ref_names:
                lines.append("  * training samples to the decoder the original run kept (best epoch x training slices; "
                             "mean over seeds)")
            lines.append("")

        # ---- validation: what the shared validation slices would pick ------------------------------------------
        lines.append(f"VALIDATION DICE (165 shared validation slices; information only, no model was chosen on it) "
                     f"and epochs per budget ({what})")
        lines.append("  " + f"{'per dataset':>11s} " + " ".join(f"{kfmt(b):>20s}" for b in budgets)
                     + f" {'val-chosen budget':>18s} {'in-domain there':>16s}")
        for n in sizes:
            cells, chosen_scores, chosen = [], [], []
            for b in budgets:
                s = grid.get((lora, n, b))
                ms_ = [metas.get((s, sd)) for sd in seeds(by, s)] if s else []
                ms_ = [m for m in ms_ if m]
                cells.append(f"{ms([m['val_dice'] for m in ms_])} ({np.mean([m['epochs_equivalent'] for m in ms_]):.0f} ep)"
                             if ms_ else "-")
            for sd in sorted({k[1] for k in metas}):
                vals = [(metas[(grid[(lora, n, b)], sd)]["val_dice"], b) for b in budgets
                        if (lora, n, b) in grid and (grid[(lora, n, b)], sd) in metas]
                if vals:
                    bb = max(vals)[1]
                    chosen.append(bb)
                    chosen_scores.append(score(by, grid[(lora, n, bb)], sd, COLS[0]))
            lines.append("  " + f"{n:>11,} " + " ".join(f"{c:>20s}" for c in cells)
                         + f" {'/'.join(kfmt(b) for b in chosen):>18s} {ms(chosen_scores):>16s}")
        lines.append("")

        for n in sizes:
            for b in budgets:
                s = grid.get((lora, n, b))
                if not s:
                    continue
                ms_ = [metas[(s, sd)] for sd in seeds(by, s) if (s, sd) in metas]
                row = {"what trains": what, "slices per dataset": n, "budget (samples)": b, "system": s,
                       "seeds": len(seeds(by, s)),
                       "training slices": int(np.mean([m["training_slices"] for m in ms_])) if ms_ else "",
                       "epochs": round(float(np.mean([m["epochs_equivalent"] for m in ms_])), 1) if ms_ else "",
                       "validation Dice": round(float(np.mean([m["val_dice"] for m in ms_])), 4) if ms_ else "",
                       "training minutes": round(float(np.mean([m["train_seconds"] for m in ms_])) / 60, 2) if ms_ else ""}
                for col in COLS:
                    v = [x for x in per_seed(by, s, col) if x == x]
                    row[f"{col[0]} mean"] = round(float(np.mean(v)), 4) if v else ""
                    row[f"{col[0]} sd"] = round(float(np.std(v, ddof=1)), 4) if len(v) > 1 else ""
                rows_csv.append(row)
                for sd in seeds(by, s):
                    m = metas.get((s, sd))
                    if not m:
                        continue
                    cpath = os.path.join(args.runs, s, f"seed{sd}", "curve.csv")
                    if os.path.isfile(cpath) and b == max(bb for (lo_, nn, bb) in grid if lo_ == lora and nn == n):
                        with open(cpath) as f:  # the longest budget's curve covers the whole run
                            for r in csv.DictReader(f):
                                val_csv.append({"what trains": what, "slices per dataset": n, "seed": sd, **r})

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write(report + "\n")
    if rows_csv:
        with open(os.path.join(args.out_dir, "fixed_grid.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
            w.writeheader()
            w.writerows(rows_csv)
    if val_csv:
        with open(os.path.join(args.out_dir, "fixed_validation.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(val_csv[0].keys()))
            w.writeheader()
            w.writerows(val_csv)
    print(f"\n-> {args.out_dir}")


if __name__ == "__main__":
    main()
