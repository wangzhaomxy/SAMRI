#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3a of the embedding experiments: layer-wise linear CKA between encoders
on IDENTICAL inputs (GPU recommended for the bootstrap; runs on CPU too).

CKA (centred kernel alignment, Kornblith et al. 2019) is 1.0 when two
representations are the same up to rotation and isotropic scaling, and near 0
when unrelated. Here it answers: how far did MedSAM's fine-tuning move the
encoder away from SAM's, and at which depth? A raw CKA value means little on
its own, so every pipeline gets three references:
  ceiling    SAM ViT-B vs SAM ViT-B with tiny input noise (--noise-std run)
  untrained  SAM ViT-B vs a randomly initialised ViT-B. NOT a floor: random
             networks largely preserve input geometry and linear CKA ignores
             rotations, so two random ViT-Bs score ~0.99 on MRI slices. It shows
             how much similarity the input structure alone produces.
  chance     SAM vs MedSAM with images shuffled (each image paired with a
             random other image, token order shuffled within the image): the
             true floor, ~0
Pairs are only formed within one pipeline, so token i of one model and token i
of the other come from the same image location.

Pairs computed automatically for each pipeline p (when both features exist):
  sam_vit_b__p      vs medsam_vit_b__p           fine-tuning drift   (main)
  sam_vit_b__p      vs sam_vit_b__p__noise<s>    noise ceiling
  sam_vit_b__p      vs medsam_vit_b__p, images shuffled   chance level
  sam_vit_b__p      vs random_vit_b__p           untrained-network reference
  medsam_vit_b__p   vs random_vit_b__p           untrained-network reference
  sam_vit_b__p      vs sam_vit_h__p              ViT-B vs ViT-H (neck only)
Token subsets: 'all' sampled tokens, and 'inbox' (inside the GT boxes, where
the prompted structure is).

Outputs in <work-dir>/results/:
  cka_results.csv   pipeline, pair, kind, layer, subset, n_images, n_tokens,
                    cka, ci_lo, ci_hi (bootstrap over images)
  cka_summary.txt   headline first, then the layer table
  cka_layers.png    by layer, one column per pipeline (needs matplotlib): CKA on
                    a 0-1 axis, and 1 - CKA on a log axis, where values close
                    to 1 (drift vs noise ceiling vs untrained) can be told apart

Usage:
  python embedding_assess/cka.py --work-dir /scratch/user/s4670484/Eval_results/embedding_research/
"""

import argparse
import csv
import json
import os
import sys
import time
from glob import glob

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emb_common as ec  # noqa: E402


class Cond:
    def __init__(self, work_dir, name):
        self.name, self.dir = name, os.path.join(work_dir, "features", name)
        self.encoder, self.pipeline, self.noise = ec.parse_cond(name)
        self.neck = np.load(os.path.join(self.dir, "neck.npy"), mmap_mode="r")
        self.neck_index = {int(r): i for i, r in enumerate(np.load(os.path.join(self.dir, "neck_rows.npy")))}
        p = os.path.join(self.dir, "layer_tokens.npy")
        self.layers = np.load(p, mmap_mode="r") if os.path.isfile(p) else None
        self.layer_names = json.load(open(os.path.join(self.dir, "layer_names.json"))) if self.layers is not None else []
        p = os.path.join(self.dir, "token_rows.npy")
        self.token_rows = np.load(p) if os.path.isfile(p) else None
        self.token_pos = np.load(os.path.join(self.dir, "token_pos.npy")) if self.token_rows is not None else None
        self.token_inbox = np.load(os.path.join(self.dir, "token_inbox.npy")) if self.token_rows is not None else None

    def tokens(self, layer):
        """(M, T, D) float32 for the CKA test rows."""
        if layer == "neck":
            out = np.empty((len(self.token_rows), self.token_pos.shape[1], 256), np.float32)
            for i, r in enumerate(self.token_rows):
                out[i] = np.asarray(self.neck[self.neck_index[int(r)]], np.float32).reshape(256, -1).T[self.token_pos[i]]
            return out
        return np.asarray(self.layers[:, self.layer_names.index(layer)], np.float32)


def linear_cka(X, Y):
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    xy = (X.T @ Y).pow(2).sum()
    return float(xy / ((X.T @ X).pow(2).sum().sqrt() * (Y.T @ Y).pow(2).sum().sqrt()))


def cka_with_ci(A, B, mask, n_boot, device, rng, shuffle_images=False):
    """A, B: (M, T, D) arrays; mask: (M, T) tokens to use. Bootstrap over images.
    shuffle_images (chance level) pairs image i of A with a random OTHER image
    of B and shuffles B's token order within each image. Both matter: the
    stored tokens are ordered in-box first, and neighbouring manifest rows
    usually come from the same dataset, so a plain shift-by-one would pair
    in-box tokens with in-box tokens of a similar image and read well above 0."""
    M, T = A.shape[:2]
    mask_b = mask
    if shuffle_images:
        order = rng.permutation(M)  # random derangement: each image -> next one in a random cycle
        pair = np.empty(M, int)
        pair[order] = order[np.roll(np.arange(M), -1)]
        tok = np.argsort(rng.random((M, T)), axis=1)
        B = np.take_along_axis(B, tok[..., None], axis=1)
        mask_b = np.take_along_axis(mask, tok, axis=1)
    else:
        pair = np.arange(M)
    At = torch.as_tensor(A, device=device)
    Bt = torch.as_tensor(B, device=device)
    Ma = torch.as_tensor(mask, device=device)
    Mb = torch.as_tensor(mask_b, device=device)
    shift = torch.as_tensor(pair, device=device)

    def one(idx):
        m = Ma[idx].reshape(-1) & Mb[shift[idx]].reshape(-1)
        return linear_cka(At[idx].reshape(-1, At.shape[-1])[m].double(),
                          Bt[shift[idx]].reshape(-1, Bt.shape[-1])[m].double())

    full = torch.arange(A.shape[0], device=device)
    est = one(full)
    boots = [one(torch.as_tensor(rng.integers(0, A.shape[0], A.shape[0]), device=device)) for _ in range(n_boot)]
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    return est, float(lo), float(hi), int((mask & mask_b[pair]).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--bootstrap", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    done = sorted(os.path.basename(os.path.dirname(p))
                  for p in glob(os.path.join(args.work_dir, "features", "*", "DONE")))
    print("finished feature sets:", ", ".join(done) or "none")

    pairs = []
    for p in ec.PIPELINES:
        sb, mb, rb, sh = (f"sam_vit_b__{p}", f"medsam_vit_b__{p}", f"random_vit_b__{p}", f"sam_vit_h__{p}")
        noisy = [c for c in done if c.startswith(f"sam_vit_b__{p}__noise")]
        cand = [(sb, mb, "fine-tuning drift (SAM vs MedSAM)", False),
                *[(sb, n, f"noise ceiling ({n.split('__')[-1]})", False) for n in noisy],
                (sb, mb, "chance (SAM vs MedSAM, images shuffled)", True),
                (sb, rb, "untrained ViT-B (SAM vs random weights)", False),
                (mb, rb, "untrained ViT-B (MedSAM vs random weights)", False),
                (sb, sh, "ViT-B vs ViT-H", False)]
        pairs += [(p, a, b, kind, sh_) for a, b, kind, sh_ in cand if a in done and b in done]
    if not pairs:
        sys.exit("No pair has both feature sets finished (look for features/<cond>/DONE).")

    out_dir = os.path.join(args.work_dir, "results")
    os.makedirs(out_dir, exist_ok=True)
    rows, cache = [], {}
    for p, a, b, kind, shuffle in pairs:
        A = cache.setdefault(a, Cond(args.work_dir, a))
        B = cache.setdefault(b, Cond(args.work_dir, b))
        if A.token_rows is None or B.token_rows is None:
            print(f"  skip {a} vs {b}: no CKA token positions stored")
            continue
        assert np.array_equal(A.token_rows, B.token_rows) and np.array_equal(A.token_pos, B.token_pos), \
            f"{a} and {b} were sampled at different rows/positions - rerun extraction with the same manifest/seed"
        common = [n for n in A.layer_names if n in B.layer_names] + ["neck"]
        print(f"{p}: {a} vs {b} [{kind}] - {len(common)} layers, {len(A.token_rows)} images", flush=True)
        for li, layer in enumerate(common):
            Xa, Xb = A.tokens(layer), B.tokens(layer)
            for subset, mask in (("all", np.ones(A.token_inbox.shape, bool)), ("inbox", A.token_inbox)):
                est, lo, hi, n = cka_with_ci(Xa, Xb, mask, args.bootstrap, device, rng, shuffle)
                rows.append({"pipeline": p, "cond_a": a, "cond_b": b, "kind": kind, "layer": layer,
                             "layer_index": li, "subset": subset, "n_images": len(A.token_rows),
                             "n_tokens": n, "cka": est, "ci_lo": lo, "ci_hi": hi})
        print(f"    done ({time.time() - t0:.0f} s)", flush=True)

    with open(os.path.join(out_dir, "cka_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- summary: headline first -------------------------------------------------
    def get(p, kind_prefix, layer, subset="all"):  # noqa: E306
        for r in rows:
            if r["pipeline"] == p and r["kind"].startswith(kind_prefix) and r["layer"] == layer and r["subset"] == subset:
                return r
        return None

    lines = ["HEADLINE: CKA between SAM ViT-B and MedSAM's encoder on identical inputs",
             "(1 = same representation up to rotation/scale. Read it against: the noise ceiling;",
             " SAM vs an UNTRAINED ViT-B, i.e. how much similarity the input structure alone produces",
             " (random networks largely preserve input geometry, so this can be high); and chance",
             " (images shuffled). Drift matters only if SAM vs MedSAM falls clearly below the ceiling.)"]
    for p in ec.PIPELINES:
        for layer in ("block11", "neck"):
            for subset in ("all", "inbox"):
                m, c, u, z = (get(p, "fine-tuning", layer, subset), get(p, "noise ceiling", layer, subset),
                              get(p, "untrained ViT-B (SAM", layer, subset), get(p, "chance", layer, subset))
                if m:
                    lines.append(f"  {p:6s} pipeline, {layer:7s}, {subset:5s} tokens: SAM vs MedSAM "
                                 f"{m['cka']:.3f} [{m['ci_lo']:.3f}, {m['ci_hi']:.3f}]"
                                 + (f" | noise ceiling {c['cka']:.3f}" if c else "")
                                 + (f" | SAM vs untrained {u['cka']:.3f}" if u else "")
                                 + (f" | chance {z['cka']:.3f}" if z else ""))
    lines += ["", "ALL LAYERS (subset 'all' | 'inbox'; 95% bootstrap CI over images)"]
    for p, a, b, kind, _ in pairs:
        lines.append(f"  [{p}] {kind}: {a} vs {b}")
        for r in rows:
            if (r["pipeline"], r["kind"], r["subset"]) == (p, kind, "all"):
                rb = next(x for x in rows if (x["pipeline"], x["kind"], x["layer"], x["subset"])
                          == (p, kind, r["layer"], "inbox"))
                lines.append(f"    {r['layer']:8s} {r['cka']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] | "
                             f"{rb['cka']:.3f} [{rb['ci_lo']:.3f}, {rb['ci_hi']:.3f}]")
    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(out_dir, "cka_summary.txt"), "w") as f:
        f.write(report + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        floor = 1e-5  # 1 - CKA is clipped here for the log axis
        fig, axes = plt.subplots(2, len(ec.PIPELINES), figsize=(13, 8), sharey="row", squeeze=False)
        kinds = list(dict.fromkeys(kind for _, _, _, kind, _ in pairs if not kind.startswith("ViT-B vs ViT-H")))
        colors = {k: f"C{i}" for i, k in enumerate(kinds)}
        for col, p in enumerate(ec.PIPELINES):
            ax_lin, ax_log = axes[0, col], axes[1, col]
            for (pp, a, b, kind, _) in pairs:
                if pp != p or kind not in colors:
                    continue
                for subset, ls in (("all", "-"), ("inbox", "--")):
                    rs = [r for r in rows if (r["pipeline"], r["kind"], r["subset"]) == (p, kind, subset)]
                    x = [r["layer_index"] for r in rs]
                    cka, lo, hi = (np.array([r[k] for r in rs]) for k in ("cka", "ci_lo", "ci_hi"))
                    kw = dict(color=colors[kind], marker=".", label=f"{kind} [{subset}]")
                    ax_lin.plot(x, cka, ls, **kw)
                    ax_lin.fill_between(x, lo, hi, color=colors[kind], alpha=0.15)
                    ax_log.plot(x, np.clip(1 - cka, floor, None), ls, **kw)
                    ax_log.fill_between(x, np.clip(1 - hi, floor, None), np.clip(1 - lo, floor, None),
                                        color=colors[kind], alpha=0.15)
                for ax in (ax_lin, ax_log):
                    ax.set_xticks(x)
                    ax.set_xticklabels([r["layer"].replace("block", "b") for r in rs], rotation=90, fontsize=7)
            ax_lin.set_title(f"{p} pipeline (identical inputs)")
            ax_lin.set_ylim(0, 1.02)
            ax_lin.set_ylabel("linear CKA")
            ax_log.set_yscale("log")
            ax_log.set_ylabel("1 - linear CKA (log; lower = more similar)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=7, loc="lower center", ncol=3)
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        fig.savefig(os.path.join(out_dir, "cka_layers.png"), dpi=130)
    except ImportError:
        pass
    print(f"\nSaved to {out_dir}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
