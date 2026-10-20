#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate the swapped-encoder decoders and the no-training references on the
in-domain and zero-shot test sets, and write the summary (headline first).

Systems
  trained arms   every <runs-dir>/<arm>/seed<k>/ with DONE (arm ids in decoder_lib.ARMS)
  references     SAM ViT-B with its original decoder (no training)
                 MedSAM end-to-end: its encoder (MedSAM preprocessing) + its own decoder
                 released SAMRI checkpoints (--samri-ckpts), on SAM-encoder features
Test sets (tight box from the native mask, one mask, Dice on native pixels)
  in-domain   test rows of --indomain-dir (11 clean datasets)
  zero-shot   test rows of --zeroshot-dir (datasets SAMRI never trained on)

Outputs in --out-dir: eval_instances.csv (one row per system x seed x structure)
and eval_summary.txt.

Usage:
  python embedding_assess/eval_decoder.py \
      --indomain-dir /scratch/user/s4670484/Eval_results/embedding_research_clean \
      --zeroshot-dir /scratch/user/s4670484/Eval_results/embedding_research_zeroshot \
      --runs-dir /scratch/user/s4670484/Eval_results/decoder_swap/runs \
      --ckpt-dir /scratch/user/s4670484/pretrained_ckpt \
      --samri-ckpts /scratch/user/s4670484/pretrained_ckpt/samri_vitb_box.pth \
                    /scratch/user/s4670484/pretrained_ckpt/samri_vitb_bp.pth \
      --out-dir /scratch/user/s4670484/Eval_results/decoder_swap/results
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from glob import glob

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decoder_lib as dl  # noqa: E402

MEANINGFUL = 0.01  # Dice; a difference must exceed this AND the seed spread


def samri_name(name):
    """Plain name for a SAMRI checkpoint file stem; every system is tested with the box prompt."""
    if name.startswith("samri_vitb_box"):
        return f"SAMRI box-prompt checkpoint {name} (36 datasets)"
    if name.startswith("samri_vitb_bp"):
        return f"SAMRI box+point checkpoint {name} (36 datasets), tested with box only"
    return f"SAMRI checkpoint {name} (36 datasets)"


def systems(args, device):
    """Yields (system id, plain name, seed, features cond, model)."""
    for run in sorted(glob(os.path.join(args.runs_dir, "*", "seed*", "DONE"))):
        run_dir = os.path.dirname(run)
        arm = os.path.basename(os.path.dirname(run_dir))
        if arm not in dl.ARMS:  # e.g. the learning-rate pilot runs
            continue
        model, meta = dl.load_trained_decoder(run_dir, args.ckpt_dir, device)
        yield arm, dl.ARMS[arm][2], meta["seed"], meta["features"], model
    yield ("ref_sam-original", "SAM ViT-B with its original decoder (no training)", 0, "sam_vit_b__sam",
           dl.build_decoder("sam", args.ckpt_dir, device, log=lambda *a: None).eval())
    yield ("ref_medsam-e2e", "MedSAM end-to-end (its encoder and its own decoder)", 0, "medsam_vit_b__medsam",
           dl.build_decoder("medsam", args.ckpt_dir, device, log=lambda *a: None).eval())
    for p in args.samri_ckpts or []:
        if os.path.isfile(p):
            name = os.path.basename(p)[:-len(".pth")]
            print(f"  {name}:")  # build_decoder reports whether its prompt encoder equals SAM's
            try:
                model = dl.build_decoder(p, args.ckpt_dir, device).eval()
            except Exception as e:  # noqa: BLE001 - one bad file must not lose the whole evaluation
                msg = " ".join(str(e).split())  # state_dict errors span several lines; keep the start of all of them
                print(f"  WARNING: could not load {p} as a SAM ViT-B checkpoint, skipped "
                      f"({type(e).__name__}: {msg[:300]}{'...' if len(msg) > 300 else ''}). "
                      f"embedding_assess/ckpt_check.py shows its contents.")
                continue
            yield (f"ref_{name}", samri_name(name), 0, "sam_vit_b__sam", model)
        else:
            print(f"  WARNING: released SAMRI checkpoint not found, skipped: {p}")


def summarise(recs):
    from scipy.stats import wilcoxon
    sys_name = {r["system"]: r["name"] for r in recs}
    test_sets = [t for t in ("in-domain", "zero-shot") if any(r["test_set"] == t for r in recs)]
    # per system/test set/seed: instance median and mean of dataset medians
    per = defaultdict(lambda: defaultdict(dict))
    for s in sys_name:
        for t in test_sets:
            for seed in sorted({r["seed"] for r in recs if r["system"] == s}):
                rs = [r for r in recs if r["system"] == s and r["test_set"] == t and r["seed"] == seed]
                if not rs:
                    continue
                by_ds = defaultdict(list)
                for r in rs:
                    by_ds[r["dataset"]].append(r["dice"])
                per[s][t][seed] = (float(np.nanmedian([r["dice"] for r in rs])),
                                   float(np.mean([np.nanmedian(v) for v in by_ds.values()])))

    order = [a for a in dl.ARMS if a in sys_name] + sorted(s for s in sys_name if s.startswith("ref_"))
    lines = ["HEADLINE - median Dice over structures | mean of per-dataset medians "
             "(trained arms: mean over seeds, +- standard deviation over seeds)"]
    for t in test_sets:
        lines.append(f"  [{t}]")
        for s in order:
            v = per[s].get(t, {})
            if not v:
                continue
            a = np.array(list(v.values()))
            sd = f" +- {a[:, 0].std(ddof=1):.3f}" if len(a) > 1 else ""
            lines.append(f"    {sys_name[s]:78s} {a[:, 0].mean():.3f}{sd:9s} | {a[:, 1].mean():.3f}  "
                         f"(seeds: {len(a)})")

    # seed-averaged dice per structure for paired comparisons
    avg = defaultdict(list)
    for r in recs:
        avg[(r["system"], r["test_set"], r["dataset"], r["row"], r["label"])].append(r["dice"])
    avg = {k: float(np.nanmean(v)) for k, v in avg.items()}

    def spread(s, t):
        v = per[s].get(t, {})
        a = [x[0] for x in v.values()]
        return (max(a) - min(a)) if len(a) > 1 else 0.0

    pairs = [("medsam-enc_sam-dec_small", "sam-enc_sam-dec_small", "Q1: does the encoder matter?"),
             ("medsam-enc_medsam-dec_small", "sam-enc_sam-dec_small", "Q1 fairness: MedSAM decoder start"),
             ("medsam-enc_medsam-dec_small", "medsam-enc_sam-dec_small", "decoder start, MedSAM encoder"),
             ("sam-enc_sam-dec_all11", "sam-enc_sam-dec_small", "Q2: more slices, same 11 datasets"),
             ("sam-enc_sam-dec_small", "ref_sam-original", "decoder training on 1.5k slices vs none")]
    for ref in sorted(x for x in sys_name if x.startswith("ref_samri_vitb_box")):  # every SAMRI box checkpoint given
        pairs += [(ref, "sam-enc_sam-dec_all11", f"Q2: more datasets (36 vs 11), {ref[4:]}"),
                  (ref, "sam-enc_sam-dec_small", f"Q2: SAMRI vs small data, {ref[4:]}"),
                  (ref, "ref_sam-original", f"SAMRI vs untrained SAM, {ref[4:]}")]
    lines += ["", f"PAIRED COMPARISONS (seed-averaged per structure; 'meaningful' = |difference| > "
                  f"{MEANINGFUL} and > seed spread)",
              "  datasets: median over datasets of the per-dataset median difference, datasets where the first "
              "is higher, Wilcoxon over datasets"]
    for a, b, what in pairs:
        if a not in sys_name or b not in sys_name:
            continue
        lines.append(f"  {what}: {sys_name[a]}  MINUS  {sys_name[b]}")
        for t in test_sets:
            keys = [k for k in avg if k[0] == a and k[1] == t]
            d_by_ds = defaultdict(list)
            for k in keys:
                kb = (b,) + k[1:]
                if kb in avg:
                    d_by_ds[k[2]].append(avg[k] - avg[kb])
            if not d_by_ds:
                continue
            ds_med = np.array([np.nanmedian(v) for v in d_by_ds.values()])
            allv = np.concatenate([np.array(v) for v in d_by_ds.values()])
            p = wilcoxon(ds_med).pvalue if len(ds_med) > 5 and np.any(ds_med != 0) else float("nan")
            diff = float(np.nanmedian(allv))
            sp = max(spread(a, t), spread(b, t))
            verdict = "meaningful" if abs(diff) > max(MEANINGFUL, sp) else "within noise"
            lines.append(f"    [{t:9s}] structures: median {diff:+.4f} (n={len(allv)}) | datasets: median "
                         f"{np.median(ds_med):+.4f}, {int((ds_med > 0).sum())}/{len(ds_med)} higher, p={p:.2g} | "
                         f"seed spread {sp:.4f} -> {verdict}")

    lines += ["", "PER DATASET (median Dice, seed-averaged)"]
    for t in test_sets:
        dss = sorted({r["dataset"] for r in recs if r["test_set"] == t})
        lines.append(f"  [{t}]  " + " | ".join(f"{s[:24]}" for s in order))
        for ds in dss:
            vals = []
            for s in order:
                v = [val for k, val in avg.items() if k[0] == s and k[1] == t and k[2] == ds]
                vals.append(f"{np.nanmedian(v):.3f}" if v else "  -  ")
            lines.append(f"    {ds:28s} " + " | ".join(f"{v:>24s}" for v in vals))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indomain-dir", required=True)
    ap.add_argument("--zeroshot-dir", default=None)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--samri-ckpts", nargs="*", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    t0 = time.time()
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    test_sets = [("in-domain", args.indomain_dir)] + ([("zero-shot", args.zeroshot_dir)] if args.zeroshot_dir else [])
    cache = {}  # (test set, features) -> (WorkDir, instances)
    recs = []
    for sid, name, seed, feats, model in systems(args, device):
        for t, wdir in test_sets:
            if (t, feats) not in cache:
                wd = dl.WorkDir(wdir, feats)
                rows = wd.rows("test")
                if t == "in-domain":  # only datasets the decoders were trained on
                    trained = {r["dataset"] for r in wd.manifest if r["split"] == "train"}
                    rows = [r for r in rows if wd.manifest[r]["dataset"] in trained]
                cache[(t, feats)] = (wd, wd.instances(rows))
            wd, inst = cache[(t, feats)]
            n0 = len(recs)
            for r, lab, d, area in dl.predict_instances(model, wd, inst, device):
                m = wd.manifest[r]
                recs.append({"system": sid, "name": name, "seed": seed, "test_set": t, "root": m["root"],
                             "dataset": m["dataset"], "row": r, "label": lab, "area": area, "dice": d})
            print(f"  {name} (seed {seed}) on {t}: median Dice "
                  f"{np.nanmedian([x['dice'] for x in recs[n0:]]):.3f} over {len(recs) - n0} structures "
                  f"({time.time() - t0:.0f} s)", flush=True)
        del model

    with open(os.path.join(args.out_dir, "eval_instances.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    report = summarise(recs)
    print("\n" + report)
    with open(os.path.join(args.out_dir, "eval_summary.txt"), "w") as f:
        f.write(report + "\n")
    runs = sorted(glob(os.path.join(args.runs_dir, "*", "seed*", "meta.json")))
    with open(os.path.join(args.out_dir, "training_runs.json"), "w") as f:
        json.dump([json.load(open(p)) for p in runs], f, indent=1)
    print(f"\nSaved to {args.out_dir}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
