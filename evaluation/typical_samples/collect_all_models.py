#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gather an all-model run of run_candidates.py: checks, the typical slice of every structure type, one npz per
slice for the figure, and preview pictures. Needs numpy, nibabel and matplotlib (sam_med2d; CPU is enough).

Rule: for every candidate, distance = the largest |Dice of this run - median Dice of the cell in the benchmark
records| over the models run; the typical slice of a structure type is its candidate with the smallest distance
(typical = distance <= 0.06).

Output in --out (the run folder):
  all_models_dice.csv   one row per candidate: Dice of every model, distance, worst model, rank in its type
  chosen.csv            one row per structure type: the candidate of rank 1
  npz/<sample_id>.npz   image (0-255), gt, one mask per model, box, point, meta (the list row and the Dice)
  png/<panel>.png       the three best candidates of the type: whole slice + one zoom per model

  python collect_all_models.py --list <typical_samples_all_models.csv> --out <run folder>
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

REPO = "/scratch/user/s4670484/SAMRI"
ORDER = ["sam_box", "medsam", "mcp_medsam", "sam_med2d", "samed", "samri_box", "samri_bp"]
NAME = {"sam_box": "SAM ViT-B", "medsam": "MedSAM", "mcp_medsam": "MCP-MedSAM", "sam_med2d": "SAM-Med2D",
        "samed": "SAMed", "samri_box": "SAMRI box", "samri_bp": "SAMRI box+point"}
COLOUR = {"sam_box": "#3D9BE0", "medsam": "#E0559F", "mcp_medsam": "#8E6BD9", "sam_med2d": "#E69F00",
          "samed": "#A0A0A0", "samri_box": "#D55E00", "samri_bp": "#12A87E"}
TYPICAL, TOL = 0.06, 0.005
BOX = ("box_x0", "box_y0", "box_x1", "box_y1")


def num(x):
    return None if x in (None, "") else float(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo-root", default=REPO)
    ap.add_argument("--top", type=int, default=3, help="candidates per picture")
    args = ap.parse_args()
    sys.path[:0] = [args.repo_root, os.path.join(args.repo_root, "evaluation", "typical_samples")]
    from typical_samples import load_slice, sam_input, zoom_window

    models = [k for k in ORDER if os.path.isfile(os.path.join(args.out, "dice_%s.csv" % k))]
    missing = [k for k in ORDER if k not in models]
    print("models run: %s%s" % (", ".join(NAME[k] for k in models),
                                ("   NOT run: " + ", ".join(NAME[k] for k in missing)) if missing else ""))
    dice = dict((k, dict((r["sample_id"], float(r["dice"])) for r in csv.DictReader(
        open(os.path.join(args.out, "dice_%s.csv" % k), newline="")))) for k in models)
    rows = [r for r in csv.DictReader(open(args.list, newline="")) if all(r["sample_id"] in dice[k] for k in models)]
    print("%d candidates with a Dice from every model run\n" % len(rows))

    print("check against the benchmark records and the v2 run (|diff| <= %.3f):" % TOL)
    for k in models:
        rec = [abs(dice[k][r["sample_id"]] - num(r["rec_" + k])) for r in rows]
        v2 = [abs(dice[k][r["sample_id"]] - num(r["v2_" + k])) for r in rows if num(r.get("v2_" + k)) is not None]
        note = "  (another checkpoint than the records: expected)" if k.startswith("samri") else ""
        print("  %-16s records %3d of %3d%s%s" % (NAME[k], sum(x <= TOL for x in rec), len(rec),
              ("   v2 run %3d of %3d" % (sum(x <= TOL for x in v2), len(v2))) if v2 else "", note))

    for r in rows:
        dev = dict((k, abs(dice[k][r["sample_id"]] - float(r["med_" + k]))) for k in models)
        r["distance"] = max(dev.values())
        r["worst_model"] = max(dev, key=dev.get)
    panels = []
    for r in rows:
        if r["panel"] not in panels:
            panels.append(r["panel"])
    for p in panels:
        rs = sorted([r for r in rows if r["panel"] == p], key=lambda r: (r["distance"], int(r["candidate"])))
        for i, r in enumerate(rs, 1):
            r["rank"] = i

    cols = ["sample_id", "panel", "candidate", "source", "rank", "distance", "worst_model"] + \
           ["dice_" + k for k in models] + ["med_" + k for k in models] + ["image", "label_value", "img_path", "seg_path"]
    out_rows = []
    for r in rows:
        o = dict((c, r.get(c)) for c in cols)
        o.update(dict(("dice_" + k, round(dice[k][r["sample_id"]], 4)) for k in models))
        o["distance"] = round(r["distance"], 4)
        out_rows.append(o)
    for name, sel in (("all_models_dice.csv", out_rows), ("chosen.csv", [o for o in out_rows if o["rank"] == 1])):
        with open(os.path.join(args.out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(sorted(sel, key=lambda o: (o["panel"], o["rank"])))

    chosen = [r for r in rows if r["rank"] == 1]
    print("\ntypical slice of every structure type (distance = largest |Dice - cell median| over %d models):" % len(models))
    print("  %-30s %4s %-9s %8s  %-11s %s" % ("structure type", "cand", "source", "distance", "worst", "  ".join(NAME[k] for k in models)))
    for r in sorted(chosen, key=lambda r: r["panel"]):
        print("  %-30s %4s %-9s %8.3f  %-11s %s%s" % (r["panel"], r["candidate"], r["source"], r["distance"],
              NAME[r["worst_model"]][:11], "  ".join("%*.2f" % (len(NAME[k]), dice[k][r["sample_id"]]) for k in models),
              "" if r["distance"] <= TYPICAL else "   (not within %.2f)" % TYPICAL))
    print("  -> %d of %d structure types have a slice within %.2f" % (sum(r["distance"] <= TYPICAL for r in chosen),
                                                                     len(chosen), TYPICAL))

    os.makedirs(os.path.join(args.out, "npz"), exist_ok=True)
    mz = dict((k, np.load(os.path.join(args.out, "masks_%s.npz" % k))) for k in models)
    items = {}
    for r in rows:
        r["label_value"] = float(r["label_value"])
        img, gt = load_slice(r)
        it = dict(image=sam_input(img)[..., 0], gt=gt, box=np.array([int(r[c]) for c in BOX]),
                  point=np.array([int(r["point_x"]), int(r["point_y"])]))
        for k in models:
            it[k] = mz[k][r["sample_id"]].astype(bool)
        meta = dict((c, r[c]) for c in r if c not in ("distance",))
        meta.update(distance=r["distance"], dice=dict((k, dice[k][r["sample_id"]]) for k in models))
        np.savez_compressed(os.path.join(args.out, "npz", r["sample_id"] + ".npz"), meta=json.dumps(meta, default=str),
                            **dict((k, v) for k, v in it.items()))
        items[r["sample_id"]] = it
    print("\nnpz saved: %d files in %s" % (len(items), os.path.join(args.out, "npz")))

    try:
        n = pictures(panels, rows, items, models, dice, args.out, args.top, zoom_window)
        print("pictures saved: %d files in %s" % (n, os.path.join(args.out, "png")))
    except Exception as e:                                   # the tables and npz are already saved
        print("!!!!! pictures failed (%s: %s); tables and npz are complete" % (type(e).__name__, e))


def pictures(panels, rows, items, models, dice, out, top, zoom_window):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    os.makedirs(os.path.join(out, "png"), exist_ok=True)
    for p in panels:
        rs = sorted([r for r in rows if r["panel"] == p], key=lambda r: r["rank"])[:top]
        ncol = 1 + len(models)
        fig, axes = plt.subplots(len(rs), ncol, figsize=(1.9 * ncol, 2.45 * len(rs) + 0.5), squeeze=False)
        for i, r in enumerate(rs):
            it = items[r["sample_id"]]
            H, W = it["gt"].shape
            xa, xb, ya, yb = zoom_window(it["box"], H, W)
            ax = axes[i][0]
            ax.imshow(it["image"], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            x0, y0, x1, y1 = [int(v) for v in it["box"]]
            ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0 + 1, y1 - y0 + 1, fill=False, ec="#35D0E0", lw=0.8))
            ax.add_patch(Rectangle((xa - 0.5, ya - 0.5), xb - xa, yb - ya, fill=False, ec="#F2D200", lw=0.8, ls="--"))
            ax.set_title("c%s (%s)  rank %d\ndistance %.3f, worst %s" % (r["candidate"], r["source"], r["rank"],
                         r["distance"], NAME[r["worst_model"]]), fontsize=6, loc="left")
            extent = (xa - 0.5, xb - 0.5, yb - 0.5, ya - 0.5)
            fill = np.zeros((yb - ya, xb - xa, 4))
            fill[it["gt"][ya:yb, xa:xb]] = (1.0, 0.83, 0.0, 0.40)
            xs, ys = np.arange(xa, xb), np.arange(ya, yb)
            for j, k in enumerate(models, 1):
                ax = axes[i][j]
                ax.imshow(it["image"][ya:yb, xa:xb], cmap="gray", vmin=0, vmax=255, interpolation="nearest", extent=extent)
                ax.imshow(fill, interpolation="nearest", extent=extent)
                mk = it[k][ya:yb, xa:xb]
                if mk.any() and not mk.all() and mk.shape[0] > 1 and mk.shape[1] > 1:
                    ax.contour(xs, ys, mk.astype(float), levels=[0.5], colors=[COLOUR[k]], linewidths=1.3)
                if k == "samri_bp":
                    ax.plot([it["point"][0]], [it["point"][1]], marker="+", ms=6, mew=1.2, color=COLOUR[k])
                ax.set_xlim(xa - 0.5, xb - 0.5)
                ax.set_ylim(yb - 0.5, ya - 0.5)
                ax.set_title("%s  %.2f" % (NAME[k], dice[k][r["sample_id"]]), fontsize=6.5, color=COLOUR[k])
        for ax in axes.ravel():
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("%s   %s   (yellow: reference mask)" % (p, rs[0]["description"]), fontsize=8, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.3 / (2.45 * len(rs) + 0.5)), h_pad=1.4)
        fig.savefig(os.path.join(out, "png", p + ".png"), dpi=130)
        plt.close(fig)
    return len(panels)


if __name__ == "__main__":
    main()
