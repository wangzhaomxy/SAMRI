#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Does training on a similar dataset help on unseen ones? (CPU, seconds)

Arms compared (eval_decoder.py output with --extra-test knee=<work dir of MSK_PD, MSK_FLASH, MSK_T2>):
  base  sam-enc_sam-dec_n<N>               the 11 clean datasets, N slices each
  plus  sam-enc_sam-dec_n<N>_plus-MSK_T2   the same 11 x N slices + N slices of MSK_T2 (knee, T2)
Seeds are paired: seed k of both arms sees the same 11-dataset slices, so plus - base isolates MSK_T2.
Test sets: unseen knee = MSK_PD + MSK_FLASH (same knee cohort and targets as MSK_T2, other contrasts);
  MSK_T2's own test split (seen by plus only); the old zero-shot sets (with and without MSK_shoulder);
  the 11 in-domain datasets (does adding MSK_T2 cost anything there?).
Metric: mean over datasets of the per-dataset median Dice, per seed; mean ± sd over seeds.
Outputs in --out-dir: knee_summary.txt (headline first), knee_all_systems.csv, knee_by_label.csv.
Several --instances files are merged (e.g. results_knee/eval_instances.csv + nnU-Net's eval_instances_*.csv).
"""

import argparse
import csv
import os
import re
from collections import defaultdict

import numpy as np

KNEE_UNSEEN = ("MSK_PD", "MSK_FLASH")
SEEN_EXTRA = "MSK_T2"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances", nargs="+", required=True,
                    help="eval_instances.csv from eval_decoder.py, plus any other instance files (e.g. nnU-Net's)")
    ap.add_argument("--knee-label", default="knee", help="test-set label used for --extra-test")
    ap.add_argument("--extra", default="MSK_T2")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    from scipy.stats import wilcoxon

    recs = []
    for path in args.instances:
        with open(path) as f:
            for r in csv.DictReader(f):
                r["seed"], r["dice"], r["label"] = int(r["seed"]), float(r["dice"]), float(r["label"])
                recs.append(r)
    by = defaultdict(list)  # (system, seed, test set, dataset) -> dice
    names = {}
    for r in recs:
        by[(r["system"], r["seed"], r["test_set"], r["dataset"])].append(r["dice"])
        names[r["system"]] = r["name"]
    systems = sorted(names)
    seeds_of = {s: sorted({k[1] for k in by if k[0] == s}) for s in systems}

    def score(system, seed, test_set, keep=None, drop=()):
        meds = [np.nanmedian(v) for (s, sd, t, ds), v in by.items()
                if s == system and sd == seed and t == test_set and (keep is None or ds in keep) and ds not in drop]
        return float(np.mean(meds)) if meds else float("nan")

    cols = [("unseen knee (PD+FLASH)", args.knee_label, KNEE_UNSEEN, ()),
            ("MSK_PD", args.knee_label, ("MSK_PD",), ()),
            ("MSK_FLASH", args.knee_label, ("MSK_FLASH",), ()),
            (f"{SEEN_EXTRA} test", args.knee_label, (SEEN_EXTRA,), ()),
            ("old zero-shot*", "zero-shot", None, ("MSK_shoulder",)),
            ("MSK_shoulder", "zero-shot", ("MSK_shoulder",), ()),
            ("in-domain (11)", "in-domain", None, ())]

    def row(system):
        out = []
        for _, t, keep, drop in cols:
            out.append([score(system, sd, t, keep, drop) for sd in seeds_of[system]])
        return out

    def fmt(v):
        v = [x for x in v if x == x]
        if not v:
            return "-"
        return f"{np.mean(v):.3f}" + (f"±{np.std(v, ddof=1):.3f}" if len(v) > 1 else "")

    def short(system):
        m = re.fullmatch(r"sam-enc_sam-dec(-lora)?_n(\d+)(?:_plus-(.+))?", system)
        if not m:
            return names[system]
        dec = "decoder LoRA" if m.group(1) else "decoder"
        return f"{dec}, {m.group(2)}/dataset, " + (f"11 datasets + {m.group(3)}" if m.group(3) else "11 datasets")

    lines = []
    # ---- headline: plus vs base ----------------------------------------------------------------
    pairs = []
    for s in systems:
        m = re.fullmatch(r"(sam-enc_sam-dec(?:-lora)?_n\d+)_plus-(.+)", s)
        if m and m.group(1) in names:
            pairs.append((m.group(1), s))
    lines.append(f"ADDING {args.extra} TO THE 11 DATASETS (same 11-dataset slices per seed; mean ± sd over seeds)")
    lines.append("  " + f"{'arm':48s}" + "".join(f"{c[0]:>24s}" for c in cols))
    diff_rows = []
    for base, plus in pairs:
        for s in (base, plus):
            lines.append("  " + f"{short(s)[:48]:48s}" + "".join(f"{fmt(v):>24s}" for v in row(s)))
        common = sorted(set(seeds_of[base]) & set(seeds_of[plus]))
        rb, rp = row(base), row(plus)
        d_cells = []
        for i, (cname, t, keep, drop) in enumerate(cols):
            db = dict(zip(seeds_of[base], rb[i]))
            dp = dict(zip(seeds_of[plus], rp[i]))
            d = [dp[k] - db[k] for k in common if dp[k] == dp[k] and db[k] == db[k]]
            pos = sum(x > 0 for x in d)
            d_cells.append(f"{np.mean(d):+.3f} ({pos}/{len(d)} seeds)" if d else "-")
            diff_rows.append({"base": base, "plus": plus, "test": cname, "mean_diff": np.mean(d) if d else "",
                              "seeds_higher": pos, "seeds": len(d)})
        lines.append("  " + f"{'  difference (plus - base)':48s}" + "".join(f"{c:>24s}" for c in d_cells))
        # structure-level paired test on the unseen knee sets (seed-averaged per structure)
        acc = defaultdict(lambda: defaultdict(list))
        for r in recs:
            if r["test_set"] == args.knee_label and r["dataset"] in KNEE_UNSEEN and r["system"] in (base, plus):
                acc[(r["dataset"], r["row"], r["label"])][r["system"]].append(r["dice"])
        dd = [np.nanmean(v[plus]) - np.nanmean(v[base]) for v in acc.values() if v[plus] and v[base]]
        if dd:
            dd = np.array(dd)
            dd = dd[~np.isnan(dd)]
            p = wilcoxon(dd).pvalue if np.any(dd != 0) else float("nan")
            lines.append(f"  {'':2s}unseen knee, per structure: median {np.median(dd):+.4f}, mean {np.mean(dd):+.4f}, "
                         f"{(dd > 0.01).mean() * 100:.0f}% better by > 0.01, {(dd < -0.01).mean() * 100:.0f}% worse "
                         f"by > 0.01 (n={len(dd)}, Wilcoxon p={p:.2g})")
        lines.append("")

    # per label (structure type) on the unseen knee sets
    lab_rows = []
    for base, plus in pairs:
        acc = defaultdict(lambda: defaultdict(list))
        for r in recs:
            if r["test_set"] == args.knee_label and r["dataset"] in KNEE_UNSEEN + (SEEN_EXTRA,) and r["system"] in (base, plus):
                acc[(r["dataset"], r["label"])][r["system"]].append(r["dice"])
        for (ds, lab), v in sorted(acc.items()):
            lab_rows.append({"base": base, "plus": plus, "dataset": ds, "label": lab, "structures_x_seeds": len(v[base]),
                             "base_median": round(float(np.nanmedian(v[base])), 4),
                             "plus_median": round(float(np.nanmedian(v[plus])), 4)})
    if lab_rows:
        lines.append("BY STRUCTURE TYPE (label value; median Dice over structures and seeds)")
        for d in lab_rows:
            lines.append(f"  {short(d['plus'])[:44]:44s} {d['dataset']:10s} label {d['label']:>6.0f}: "
                         f"base {d['base_median']:.3f} -> plus {d['plus_median']:.3f} ({d['plus_median'] - d['base_median']:+.3f})")
        lines.append("")

    # ---- every system on the knee sets ------------------------------------------------------------
    lines.append("EVERY SYSTEM ON THE KNEE SETS (sorted by unseen knee Dice)")
    show = cols[:4] + cols[5:]  # knee sets, MSK_shoulder, in-domain
    lines.append("  " + f"{'system':78s}" + "".join(f"{c[0]:>24s}" for c in show))
    allrows = []
    for s in systems:
        r = row(s)
        allrows.append((np.nanmean(r[0]) if r[0] else np.nan, s, r))
    for key, s, r in sorted(allrows, key=lambda x: -x[0] if x[0] == x[0] else 9):
        lines.append("  " + f"{names[s][:78]:78s}" + "".join(f"{fmt(v):>24s}" for v in r[:4] + r[5:]))
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(args.out_dir, "knee_summary.txt"), "w") as f:
        f.write(report + "\n")
    with open(os.path.join(args.out_dir, "knee_all_systems.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "name", "seeds"] + [c[0] for c in cols])
        for _, s, r in allrows:
            w.writerow([s, names[s], len(seeds_of[s])] + [np.nanmean(v) if any(x == x for x in v) else "" for v in r])
    with open(os.path.join(args.out_dir, "knee_by_label.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["base", "plus", "dataset", "label", "structures_x_seeds", "base_median", "plus_median"])
        w.writeheader()
        w.writerows(lab_rows)


if __name__ == "__main__":
    main()
