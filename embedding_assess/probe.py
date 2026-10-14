#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3b of the embedding experiments: does a FROZEN embedding already contain
what box-prompted MRI segmentation needs? (GPU recommended; CPU works.)

A promptable model is told WHERE the target is (the box), so the encoder only
has to make the target separable from its surroundings inside the box; it does
not have to know WHAT the structure is. The probes therefore get the box but
no class label, and the same tiny heads are trained on every encoder's
embedding, so differences come from the embeddings alone.

For each condition (encoder x pipeline, plus raw-pixel baselines) and each
instance (slice x labelled structure), on the tokens inside the GT box
enlarged by one token:
  auc_proto    no training: split the box tokens in two random halves, build
               target/background prototypes from one half, score the other
               half by cosine similarity (and swap); token-level ROC AUC.
  linear, mlp  one probe shared by ALL datasets (no class labels), trained on
               the train rows: input = 256-d token + 4 box-relative position
               features; output = "is this token inside the target?".
               Scored as token AUC and as Dice on NATIVE pixels after mapping
               the 64x64 probability map back the way SAM / MedSAM map masks.
  dice_ceiling the best any 64x64-token prediction can do: the true token
               coverage mapped back the same way. Report Dice / ceiling.
  dice_box     Dice of the filled box itself (what the prompt alone gives).

Conditions (default: every finished, noise-free features/<cond>/ with the neck
stored for all rows, plus pixels__sam and pixels__medsam):
  pixels__<p>  the 16x16 pixel patch behind each token (256 values), i.e. what
               a probe can do with no encoder at all.

Outputs in <work-dir>/results/:
  probe_instances.csv  one row per (condition, test instance)
  probe_summary.txt    headline first (per condition, trained vs zero-shot
                       datasets), paired comparisons, then per dataset

Usage:
  python embedding_assess/probe.py --work-dir /scratch/user/s4670484/Eval_results/embedding_research/
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
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emb_common as ec  # noqa: E402


def auc(scores, labels):
    labels = np.asarray(labels, bool)
    npos, nneg = labels.sum(), (~labels).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float)
    s = np.asarray(scores)[order]
    i = 0
    while i < len(s):  # average ranks over ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return float((ranks[labels].sum() - npos * (npos + 1) / 2) / (npos * nneg))


class Features:
    """Per-row token features (64, 64, C) for one condition."""

    def __init__(self, work_dir, cond, manifest):
        self.cond, self.manifest = cond, manifest
        self.pipeline = cond.split("__")[1]
        if cond.startswith("pixels__"):
            self.neck = None
        else:
            d = os.path.join(work_dir, "features", cond)
            self.neck = np.load(os.path.join(d, "neck.npy"), mmap_mode="r")
            self.index = {int(r): i for i, r in enumerate(np.load(os.path.join(d, "neck_rows.npy")))}

    def has_row(self, r):
        return self.neck is None or r in self.index

    def __call__(self, r):
        if self.neck is not None:
            return np.asarray(self.neck[self.index[r]], np.float32).transpose(1, 2, 0)
        x, _ = ec.preprocess(ec.load_slice(self.manifest[r]["img_path"]), self.pipeline)
        x0 = x[0].numpy()
        return x0.reshape(ec.GRID, ec.PATCH, ec.GRID, ec.PATCH).transpose(0, 2, 1, 3).reshape(ec.GRID, ec.GRID, -1)


def instance_tokens(feat_map, geo, k, valid):
    """Token features, labels, position features and grid indices for one
    instance: tokens in the GT box enlarged by one token, valid area only."""
    gx0, gy0, gx1, gy1 = [int(v) for v in geo["inst_boxgrid"][k]]
    ys, xs = np.mgrid[max(gy0 - 1, 0):min(gy1 + 2, ec.GRID), max(gx0 - 1, 0):min(gx1 + 2, ec.GRID)]
    keep = valid[ys, xs]
    ys, xs = ys[keep], xs[keep]
    cov = geo["inst_cov"][k][ys, xs] / 255.0
    H, W, h, w = geo["row_geom"][geo["inst_row"][k]]
    b = ec.box_to_input(geo["inst_box"][k], (H, W, h, w)) / ec.PATCH  # box in token units (float)
    cx, cy = (b[0] + b[2] + 1) / 2, (b[1] + b[3] + 1) / 2
    hw, hh = max((b[2] - b[0] + 1) / 2, 0.5), max((b[3] - b[1] + 1) / 2, 0.5)
    u, v = (xs + 0.5 - cx) / hw, (ys + 0.5 - cy) / hh
    inside = ((xs >= gx0) & (xs <= gx1) & (ys >= gy0) & (ys <= gy1)).astype(np.float32)
    pos = np.stack([u, v, u * u + v * v, inside], 1).astype(np.float32)
    return feat_map[ys, xs], cov >= 0.5, pos, ys, xs


def proto_auc(F_tok, y, rng):
    """Cross-fitted prototype separability (no training)."""
    if y.sum() < 2 or (~y).sum() < 2:
        return float("nan")
    Fn = F_tok / (np.linalg.norm(F_tok, axis=1, keepdims=True) + 1e-8)
    half = rng.permutation(len(y)) < len(y) // 2
    out = []
    for a, b in ((half, ~half), (~half, half)):
        if y[a].sum() == 0 or (~y[a]).sum() == 0:
            continue
        pf, pb = Fn[a & y].mean(0), Fn[a & ~y].mean(0)
        s = Fn[b] @ (pf / (np.linalg.norm(pf) + 1e-8)) - Fn[b] @ (pb / (np.linalg.norm(pb) + 1e-8))
        out.append(auc(s, y[b]))
    return float(np.nanmean(out)) if out else float("nan")


def train_probe(kind, X, y, device, epochs, seed):
    torch.manual_seed(seed)
    d = X.shape[1]
    net = nn.Linear(d, 1) if kind == "linear" else nn.Sequential(
        nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
    net = net.to(device)
    Xt, yt = torch.as_tensor(X, device=device), torch.as_tensor(y, dtype=torch.float32, device=device)
    pw = torch.tensor([(len(y) - y.sum()) / max(y.sum(), 1)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(epochs):
        for idx in torch.randperm(len(y), generator=g).split(4096):
            idx = idx.to(device)
            opt.zero_grad()
            loss_fn(net(Xt[idx]).squeeze(1), yt[idx]).backward()
            opt.step()
    return net.eval()


def run_condition(cond, args, manifest, device):
    t0 = time.time()
    F = Features(args.work_dir, cond, manifest)
    geo = np.load(os.path.join(args.work_dir, f"geometry_{F.pipeline}.npz"))
    inst_by_row = defaultdict(list)
    for k, r in enumerate(geo["inst_row"]):
        inst_by_row[int(r)].append(k)
    rng = np.random.default_rng(args.seed)

    train_X, train_y, test = [], [], []
    for r in sorted(inst_by_row):
        if not F.has_row(r):
            continue
        fmap, valid = F(r), geo["row_valid"][r]
        for k in inst_by_row[r]:
            ft, y, pos, ys, xs = instance_tokens(fmap, geo, k, valid)
            if manifest[r]["split"] == "train":
                train_X.append(np.concatenate([ft, pos], 1))
                train_y.append(y)
            else:
                test.append({"k": k, "row": r, "ft": ft, "pos": pos, "y": y, "ys": ys, "xs": xs})
    X = np.concatenate(train_X)
    Y = np.concatenate(train_y)
    if len(Y) > args.max_train_tokens:
        sel = rng.choice(len(Y), args.max_train_tokens, replace=False)
        X, Y = X[sel], Y[sel]
    C = X.shape[1] - 4
    mu, sd = X[:, :C].mean(0), X[:, :C].std(0) + 1e-6  # standardise the embedding dims (train tokens)
    X[:, :C] = (X[:, :C] - mu) / sd
    print(f"  {cond}: {len(Y)} train tokens ({Y.mean():.1%} target), {len(test)} test instances, "
          f"C={C} ({time.time() - t0:.0f} s to load)", flush=True)

    nets = {kind: train_probe(kind, X.astype(np.float32), Y, device, args.epochs, args.seed)
            for kind in ("linear", "mlp")}

    out, seg_cache = [], {}
    for t in test:
        r, k = t["row"], t["k"]
        if r not in seg_cache:
            seg_cache = {r: ec.load_slice(manifest[r]["img_path"], manifest[r]["seg_path"])[1]}
        m = seg_cache[r] == geo["inst_label"][k]
        geom = tuple(int(v) for v in geo["row_geom"][r])
        ftz = (t["ft"] - mu) / sd
        rec = {"cond": cond, "pipeline": F.pipeline, "encoder": cond.split("__")[0],
               "root": manifest[r]["root"], "dataset": manifest[r]["dataset"],
               "zeroshot": manifest[r]["zeroshot"], "row": r, "label": float(geo["inst_label"][k]),
               "area": int(geo["inst_area"][k]), "n_tokens": len(t["y"]), "frac_target": float(t["y"].mean()),
               "auc_proto": proto_auc(ftz, t["y"], np.random.default_rng([args.seed, k]))}
        Xi = torch.as_tensor(np.concatenate([ftz, t["pos"]], 1).astype(np.float32), device=device)
        for kind, net in nets.items():
            with torch.no_grad():
                p = torch.sigmoid(net(Xi).squeeze(1)).cpu().numpy()
            grid = np.zeros((ec.GRID, ec.GRID), np.float32)
            grid[t["ys"], t["xs"]] = p
            rec[f"auc_{kind}"] = auc(p, t["y"])
            rec[f"dice_{kind}"] = ec.dice(ec.grid_to_native(grid, geom) > 0.5, m)
        rec["dice_ceiling"] = ec.dice(ec.grid_to_native(geo["inst_cov"][k] / 255.0, geom) > 0.5, m)
        x0, y0, x1, y1 = [int(v) for v in geo["inst_box"][k]]
        boxm = np.zeros_like(m)
        boxm[y0:y1 + 1, x0:x1 + 1] = True
        rec["dice_box"] = ec.dice(boxm, m)
        out.append(rec)
    print(f"    done in {time.time() - t0:.0f} s", flush=True)
    return out


def summarise(recs, conds):
    from scipy.stats import wilcoxon
    med = lambda v: float(np.nanmedian(v)) if len(v) else float("nan")  # noqa: E731
    mean = lambda v: float(np.nanmean(v)) if len(v) else float("nan")  # noqa: E731
    by = defaultdict(list)
    for r in recs:
        by[r["cond"]].append(r)
    lines = ["HEADLINE (test instances; trained datasets | zero-shot datasets)",
             f"  {'condition':28s} {'n':>5s}  {'proto AUC':>15s}  {'linear AUC':>15s}  {'MLP AUC':>15s}  "
             f"{'MLP Dice':>15s}  {'MLP Dice/ceiling':>17s}"]
    for c in conds:
        rs = by[c]
        parts = []
        for key in ("auc_proto", "auc_linear", "auc_mlp", "dice_mlp"):
            a = [r[key] for r in rs if not r["zeroshot"]]
            b = [r[key] for r in rs if r["zeroshot"]]
            f = mean if key.startswith("auc") else med
            parts.append(f"{f(a):.3f} | {f(b):.3f}")
        ratio = [r["dice_mlp"] / r["dice_ceiling"] for r in rs if r["dice_ceiling"] and r["dice_ceiling"] > 0]
        lines.append(f"  {c:28s} {len(rs):5d}  " + "  ".join(f"{p:>15s}" for p in parts) + f"  {med(ratio):17.3f}")
    for p in ec.PIPELINES:  # references depend only on the pipeline's geometry
        rs = next((by[c] for c in conds if c.split("__")[1] == p and by[c]), None)
        if rs:
            lines.append(f"  {p} pipeline references: median Dice ceiling (64x64 token resolution) "
                         f"{med([r['dice_ceiling'] for r in rs]):.3f}, box-only Dice "
                         f"{med([r['dice_box'] for r in rs]):.3f}")
    lines.append("  AUC columns are means, Dice columns medians.")

    def wil(d, min_n):
        return wilcoxon(d).pvalue if len(d) >= min_n and np.any(d != 0) else float("nan")

    lines += ["", "PAIRED COMPARISONS (other minus base, same instances)",
              "  instances: median difference, Wilcoxon p over instances (optimistic: structures of one",
              "             slice / dataset are not independent)",
              "  datasets:  per-dataset median of the instance differences; median over datasets, number of",
              "             datasets where other > base, Wilcoxon p over datasets (the honest test)"]
    comparisons = [(f"sam_vit_b__{p}", o.format(p=p)) for p in ec.PIPELINES
                   for o in ("medsam_vit_b__{p}", "sam_vit_h__{p}", "random_vit_b__{p}", "pixels__{p}")]
    comparisons.append(("sam_vit_b__sam", "medsam_vit_b__medsam"))  # each encoder as deployed
    for base, other in comparisons:
        if base not in by or other not in by:
            continue
        if base.split("__")[1] != other.split("__")[1]:
            lines.append(f"  -- each in its own pipeline (SAMRI's encoder as SAMRI uses it vs MedSAM as MedSAM uses it):")
        A = {(r["row"], r["label"]): r for r in by[base]}
        B = {(r["row"], r["label"]): r for r in by[other]}
        keys = sorted(set(A) & set(B))
        for metric in ("auc_proto", "auc_mlp", "dice_mlp"):
            d_by_ds = defaultdict(list)
            for k in keys:
                d = B[k][metric] - A[k][metric]
                if not np.isnan(d):
                    d_by_ds[(A[k]["root"], A[k]["dataset"])].append(d)
            d = np.concatenate([v for v in d_by_ds.values()]) if d_by_ds else np.zeros(0)
            dd = np.array([np.median(v) for v in d_by_ds.values()])
            lines.append(f"  {other:24s} minus {base:16s} {metric:9s}: "
                         f"instances median {med(d):+.4f} (n={len(d)}, p={wil(d, 11):.2g}) | "
                         f"datasets median {med(dd):+.4f} ({int((dd > 0).sum())}/{len(dd)} higher, "
                         f"p={wil(dd, 6):.2g})")

    lines += ["", "PER DATASET (median MLP Dice; columns = conditions)"]
    datasets = sorted({(r["root"], r["dataset"]) for r in recs})
    lines.append(f"  {'dataset':40s} " + " ".join(f"{c[:22]:>22s}" for c in conds))
    for root, ds in datasets:
        vals = [med([r["dice_mlp"] for r in by[c] if (r["root"], r["dataset"]) == (root, ds)]) for c in conds]
        lines.append(f"  {root + '/' + ds:40s} " + " ".join(f"{v:22.3f}" for v in vals))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--max-train-tokens", type=int, default=3_000_000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    device = torch.device(args.device)
    manifest = ec.read_manifest(os.path.join(args.work_dir, "manifest.csv"))

    if args.conditions:
        conds = args.conditions
    else:
        conds = []
        for p in sorted(glob(os.path.join(args.work_dir, "features", "*", "DONE"))):
            d = os.path.dirname(p)
            meta = json.load(open(os.path.join(d, "meta.json")))
            if not meta["noise_std"] and meta["n_neck_rows"] == len(manifest):
                conds.append(os.path.basename(d))
        conds += [f"pixels__{p}" for p in ec.PIPELINES]
    print("conditions:", ", ".join(conds), flush=True)

    recs = []
    for c in conds:
        recs += run_condition(c, args, manifest, device)
    out_dir = os.path.join(args.work_dir, "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "probe_instances.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    report = summarise(recs, conds)
    print("\n" + report)
    with open(os.path.join(out_dir, "probe_summary.txt"), "w") as f:
        f.write(report + "\n")
    print(f"\nSaved to {out_dir}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
