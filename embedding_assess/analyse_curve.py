#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tables for the data-size, data-variety and compute questions (CPU, seconds).

Inputs
  --instances   eval_instances.csv files (eval_decoder.py outputs; image-trained arms are already merged
                in when eval_decoder.py was given --extra-instances). Rows are de-duplicated by
                (system, seed, test set, row, label).
  --runs        run roots holding <arm>/seed<k>/meta.json (decoder_swap/runs, runs_curve, runs_lora);
                learning-rate trial folders (<arm>_lr...) are ignored, as in eval_decoder.py.
  --pool-json   runs_curve/pool.json (encoding time per slice, used to charge decoder-only arms trained on
                stored features for the one-off encoding they rely on).
Metric: mean over datasets of the per-dataset median Dice (as in eval_summary.txt), per seed; then mean
  and standard deviation over seeds (or over draws for the variety runs).
Outputs in --out-dir: data_size.csv, variety.csv, compute.csv and summary.txt (headline first).
"""

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from glob import glob

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoder_lib import IMAGE_ARMS  # noqa: E402

ZS_HARD = "MSK_shoulder"  # reported separately: every trained decoder collapses there


def load_instances(paths):
    seen, recs = set(), []
    for p in paths:
        with open(p) as f:
            for r in csv.DictReader(f):
                k = (r["system"], r["seed"], r["test_set"], r["row"], r["label"])
                if k in seen:
                    continue
                seen.add(k)
                r["seed"], r["dice"] = int(r["seed"]), float(r["dice"])
                recs.append(r)
    return recs


def load_metas(roots):
    metas = {}
    for root in roots:
        for p in glob(os.path.join(root, "*", "seed*", "meta.json")):
            arm = os.path.basename(os.path.dirname(os.path.dirname(p)))
            if re.search(r"_lr[0-9e.\-]+$|_pilot", arm):
                continue
            m = json.load(open(p))
            metas[(arm, int(m.get("seed", 0)))] = m
    return metas


def score(rows, keep_ds=None, drop_ds=None):
    by_ds = defaultdict(list)
    for r in rows:
        if keep_ds is not None and r["dataset"] not in keep_ds:
            continue
        if drop_ds is not None and r["dataset"] in drop_ds:
            continue
        by_ds[r["dataset"]].append(r["dice"])
    return float(np.mean([np.nanmedian(v) for v in by_ds.values()])) if by_ds else float("nan")


def ms(v):
    v = [x for x in v if x == x]
    if not v:
        return "  -  "
    return f"{np.mean(v):.3f}" + (f" ± {np.std(v, ddof=1):.3f}" if len(v) > 1 else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances", nargs="+", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--pool-json", default=None)
    ap.add_argument("--small-slices", type=int, default=1485, help="training slices of the manifest 'small' arms")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    recs = load_instances(args.instances)
    metas = load_metas(args.runs)
    enc_per_slice = json.load(open(args.pool_json))["encode_seconds_per_slice"] if args.pool_json else None

    rows = defaultdict(list)  # (system, seed, test set) -> records
    names = {}
    for r in recs:
        rows[(r["system"], r["seed"], r["test_set"])].append(r)
        names[r["system"]] = r["name"]
    systems = sorted({k[0] for k in rows})
    in_domain_ds = sorted({r["dataset"] for r in recs if r["test_set"] == "in-domain"})

    def per_seed(system, test_set, **kw):
        return [score(rows[(system, s, test_set)], **kw) for s in sorted({k[1] for k in rows if k[0] == system})
                if rows.get((system, s, test_set))]

    lines = []
    # ---- data size ---------------------------------------------------------------------------
    size_rows = []
    for system in systems:
        m = re.fullmatch(r"(sam-enc_sam-dec(?:-lora)?)_n(\d+)", system)
        if not m:
            continue
        seeds = sorted({k[1] for k in rows if k[0] == system})
        ms_ = [metas.get((system, s), {}) for s in seeds]
        size_rows.append({"what": "LoRA on decoder" if m.group(1).endswith("lora") else "decoder trained",
                          "per_dataset": int(m.group(2)),
                          "training_slices": int(np.mean([x.get("training_slices", 0) for x in ms_])) if ms_ else 0,
                          "seeds": len(seeds),
                          "in_domain": per_seed(system, "in-domain"), "zero_shot": per_seed(system, "zero-shot"),
                          "zero_shot_wo_msk": per_seed(system, "zero-shot", drop_ds={ZS_HARD}),
                          "gpu_min": float(np.mean([x.get("seconds", np.nan) for x in ms_])) / 60 if ms_ else np.nan})
    size_rows.sort(key=lambda d: (d["what"], d["per_dataset"]))
    with open(os.path.join(args.out_dir, "data_size.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["what trains", "slices per dataset", "training slices", "seeds", "in-domain mean", "in-domain sd",
                    "zero-shot mean", "zero-shot sd", "zero-shot without MSK_shoulder", "GPU minutes"])
        for d in size_rows:
            w.writerow([d["what"], d["per_dataset"], d["training_slices"], d["seeds"],
                        np.mean(d["in_domain"]), np.std(d["in_domain"], ddof=1) if len(d["in_domain"]) > 1 else "",
                        np.mean(d["zero_shot"]), np.std(d["zero_shot"], ddof=1) if len(d["zero_shot"]) > 1 else "",
                        np.mean(d["zero_shot_wo_msk"]), round(d["gpu_min"], 1)])

    # ---- data variety -------------------------------------------------------------------------
    var = defaultdict(list)
    var_rows = []
    for system in systems:
        m = re.fullmatch(r"sam-enc_sam-dec_k(\d+)-draw(\d+)", system)
        if not m:
            continue
        k = int(m.group(1))
        meta = metas.get((system, 0), {})
        train_ds = set(meta.get("datasets", []))
        d = {"k": k, "draw": int(m.group(2)), "datasets": " ".join(sorted(train_ds)),
             "seen": score(rows[(system, 0, "in-domain")], keep_ds=train_ds),
             "unseen_clean": score(rows[(system, 0, "in-domain")], drop_ds=train_ds),
             "zero_shot": score(rows[(system, 0, "zero-shot")]),
             "zero_shot_wo_msk": score(rows[(system, 0, "zero-shot")], drop_ds={ZS_HARD})}
        var_rows.append(d)
        var[k].append(d)
    with open(os.path.join(args.out_dir, "variety.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["k", "draw", "datasets", "seen", "unseen_clean", "zero_shot", "zero_shot_wo_msk"])
        w.writeheader()
        w.writerows(sorted(var_rows, key=lambda d: (d["k"], d["draw"])))

    # ---- compute ------------------------------------------------------------------------------
    comp = []
    for system in systems:
        seeds = sorted({k[1] for k in rows if k[0] == system})
        ms_ = [metas[(system, s)] for s in seeds if (system, s) in metas]
        if not ms_:
            continue
        secs = []
        for x in ms_:
            if x.get("arm", system) in IMAGE_ARMS or system in IMAGE_ARMS:  # image-trained arm (train_lora.py)
                # train_seconds from the current train_lora.py; older runs only have seconds (incl. the test)
                secs.append(x.get("train_seconds", x.get("seconds", np.nan)))
            elif x.get("source") == "manifest":  # stored features: charge the one-off encoding as well
                n = x.get("training_slices", args.small_slices)
                secs.append(x["seconds"] + (n * enc_per_slice if enc_per_slice else 0))
            else:
                secs.append(x.get("seconds", np.nan))
        comp.append({"system": system, "name": names.get(system, system), "seeds": len(seeds),
                     "gpu_hours": float(np.nanmean(secs)) / 3600,
                     "trainable_parameters": ms_[0].get("trainable_parameters", ms_[0].get("lora_parameters", "")),
                     "peak_gpu_gb": ms_[0].get("peak_gpu_gb", ms_[0].get("peak_gpu_gb_training", "")),
                     "in_domain": float(np.mean(per_seed(system, "in-domain") or [np.nan])),
                     "zero_shot_wo_msk": float(np.mean(per_seed(system, "zero-shot", drop_ds={ZS_HARD}) or [np.nan]))})
    comp.sort(key=lambda d: d["gpu_hours"])
    with open(os.path.join(args.out_dir, "compute.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(comp[0].keys()) if comp else ["system"])
        w.writeheader()
        w.writerows(comp)

    # ---- summary ------------------------------------------------------------------------------
    lines.append("COMPUTE vs ACCURACY (cheapest first; GPU-hours incl. the one-off encoding for decoder-only arms)")
    lines.append(f"  {'system':86s} {'GPU-h':>7s} {'params':>10s} {'in-domain':>10s} {'zero-shot*':>10s}")
    for d in comp:
        p = d["trainable_parameters"]
        p = f"{p / 1e6:.2f} M" if isinstance(p, (int, float)) and p else ""
        lines.append(f"  {d['name'][:86]:86s} {d['gpu_hours']:7.2f} {p:>10s} {d['in_domain']:10.3f} "
                     f"{d['zero_shot_wo_msk']:10.3f}")
    lines.append(f"  * zero-shot without {ZS_HARD}")
    lines.append("")
    lines.append("DATA SIZE (decoder-only; equal slices per dataset; mean ± sd over seeds)")
    lines.append(f"  {'what trains':18s} {'per dataset':>11s} {'slices':>7s} {'in-domain':>15s} {'zero-shot':>15s} "
                 f"{'zero-shot*':>15s} {'GPU-min':>8s}")
    for d in size_rows:
        lines.append(f"  {d['what']:18s} {d['per_dataset']:11d} {d['training_slices']:7d} {ms(d['in_domain']):>15s} "
                     f"{ms(d['zero_shot']):>15s} {ms(d['zero_shot_wo_msk']):>15s} {d['gpu_min']:8.1f}")
    lines.append("")
    lines.append(f"DATA VARIETY (decoder trained; same total slices; mean ± sd over draws)")
    lines.append("  seen = test structures of the datasets trained on; unseen clean = the clean datasets left out")
    lines.append(f"  {'datasets':>8s} {'seen':>15s} {'unseen clean':>15s} {'zero-shot':>15s} {'zero-shot*':>15s}")
    for k in sorted(var):
        v = var[k]
        lines.append(f"  {k:8d} {ms([x['seen'] for x in v]):>15s} {ms([x['unseen_clean'] for x in v]):>15s} "
                     f"{ms([x['zero_shot'] for x in v]):>15s} {ms([x['zero_shot_wo_msk'] for x in v]):>15s}")
    lines.append(f"  (in-domain test datasets: {len(in_domain_ds)})")
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
