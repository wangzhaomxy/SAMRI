#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot layer-wise CKA from results/cka_results.csv (written by cka.py).

2 x 2 small multiples: columns = preprocessing (SAM's / MedSAM's), rows = token
subset (all sampled tokens / tokens inside the target boxes). Each panel shows
SAM ViT-B vs MedSAM's encoder (with 95% bootstrap band), SAM ViT-B vs an
untrained ViT-B, the noise ceiling and chance level; the SAM-preprocessing
panels also mark SAM ViT-B vs ViT-H at the neck.

Usage:
  python embedding_assess/plot_cka.py --results <work-dir>/results/cka_results.csv
Writes cka_layers.png and cka_layers.pdf next to the CSV (or to --out-dir).
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3 (validated)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = [  # (kind prefix in cka_results.csv, legend label, colour)
    ("fine-tuning drift", "SAM ViT-B vs MedSAM encoder", BLUE),
    ("untrained ViT-B (SAM", "SAM ViT-B vs untrained ViT-B", ORANGE),
]


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k in ("cka", "ci_lo", "ci_hi"):
            r[k] = float(r[k])
        r["layer_index"] = int(r["layer_index"])
    return rows


def series(rows, pipeline, kind, subset):
    rs = sorted((r for r in rows if r["pipeline"] == pipeline and r["kind"].startswith(kind)
                 and r["subset"] == subset), key=lambda r: r["layer_index"])
    return rs


def place_end_labels(ax, x_end, items, min_gap=0.055):
    """items: list of (y, text, colour). Spread labels vertically so none overlap."""
    items = sorted(items, key=lambda t: t[0])
    ys = [t[0] for t in items]
    for k in range(1, len(ys)):
        ys[k] = max(ys[k], ys[k - 1] + min_gap)
    for (y0, txt, col), y in zip(items, ys):
        ax.text(x_end + 0.3, y, txt, color=col, va="center", fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    rows = load(args.results)
    out = args.out_dir or os.path.dirname(os.path.abspath(args.results))

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.6), sharey=True, sharex=True)
    pipelines = [("sam", "SAM preprocessing (resize + pad)"), ("medsam", "MedSAM preprocessing (clip + stretch)")]
    subsets = [("all", "all sampled tokens"), ("inbox", "tokens inside target boxes")]

    for i, (subset, slabel) in enumerate(subsets):
        for j, (pl, plabel) in enumerate(pipelines):
            ax = axes[i, j]
            ends = []
            ref = series(rows, pl, "fine-tuning drift", subset)
            if not ref:
                ax.set_visible(False)
                continue
            x = [r["layer_index"] for r in ref]
            labels = ["patch\nemb."] + [str(k) for k in range(1, len(x) - 1)] + ["neck"]
            ceil = series(rows, pl, "noise ceiling", subset)
            if ceil:
                ax.plot(x, [r["cka"] for r in ceil], color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=2)
                ends.append((ceil[-1]["cka"], "noise ceiling", INK2))
            chance = series(rows, pl, "chance", subset)
            if chance:
                ax.plot(x, [r["cka"] for r in chance], color=INK2, lw=1.0, ls=(0, (1, 2.5)), zorder=2)
                ends.append((chance[-1]["cka"], "chance", INK2))
            for kind, lab, col in SERIES:
                rs = series(rows, pl, kind, subset)
                if not rs:
                    continue
                ax.fill_between(x, [r["ci_lo"] for r in rs], [r["ci_hi"] for r in rs], color=col, alpha=0.18,
                                lw=0, zorder=3)
                ax.plot(x, [r["cka"] for r in rs], color=col, lw=2, marker="o", ms=4,
                        markeredgecolor="white", markeredgewidth=0.8, zorder=4, label=lab)
                ends.append((rs[-1]["cka"], f"{rs[-1]['cka']:.2f}", INK))
            h = series(rows, pl, "ViT-B vs ViT-H", subset)
            if h:
                ax.plot([x[-1]], [h[0]["cka"]], marker="D", ms=7, ls="none", color=AQUA, markeredgecolor="white",
                        markeredgewidth=1.0, zorder=5, label="SAM ViT-B vs SAM ViT-H (neck only)")
                ax.annotate(f"ViT-H {h[0]['cka']:.2f}", (x[-1], h[0]["cka"]), xytext=(-8, -2),
                            textcoords="offset points", ha="right", va="center", fontsize=8, color=INK)
            place_end_labels(ax, x[-1], ends)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=7.5)
            ax.set_xlim(x[0] - 0.4, x[-1] + 2.6)
            ax.set_ylim(-0.03, 1.05)
            ax.grid(axis="y", color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if i == 0:
                ax.set_title(plabel, fontsize=10, color=INK, pad=6)
            if j == 0:
                ax.set_ylabel(f"linear CKA\n({slabel})")
            if i == 1:
                ax.set_xlabel("encoder layer (transformer block)")

    handles, labs = [], []
    for ax in axes.ravel():
        for hnd, lb in zip(*ax.get_legend_handles_labels()):
            if lb not in labs:
                handles.append(hnd)
                labs.append(lb)
    fig.legend(handles, labs, loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("How far MedSAM's fine-tuning moved SAM's image encoder (identical inputs, 623 test slices)",
                 fontsize=11, y=0.995)
    fig.text(0.5, 0.005, "Shaded bands: 95% bootstrap intervals over images. Noise ceiling: SAM ViT-B vs itself with "
             "small input noise. Chance: images shuffled.", ha="center", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.025, 1, 0.92))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"cka_layers.{ext}"), dpi=200)
    print("saved", os.path.join(out, "cka_layers.png"), "and .pdf")


if __name__ == "__main__":
    main()
