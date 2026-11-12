#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
What do the decoders actually predict on the structures that collapse? (1 GPU, ~10 min; CPU works, slower)

eval_instances.csv keeps only Dice. This script reruns the same prediction (same features, same tight box,
same thresholding as eval_decoder.py / train_lora.py) for a few systems and keeps the masks, so a Dice of 0
can be told apart: an empty prediction, the prompt box filled with something else, another labelled
structure (knee: femur / tibia bone are labelled), or unlabelled tissue (shoulder: bone is not labelled).

Systems (seed 0 where there are seeds): untrained SAM; decoder trained, small data; decoder LoRA, small data;
decoder trained, 500 per dataset; the same + MSK_T2; LoRA on encoder and decoder, small data.
Targets: MSK_shoulder scapula cartilage (180) and humerus cartilage (225), zero-shot set; MSK_FLASH femoral
cartilage (2) and tibial cartilage (4), knee set.
Per instance: Dice (must equal eval_instances.csv - printed as a check), GT area, predicted area, share of the
prediction inside the prompt box, share of the box filled, share of the prediction on the target, on each
other label, on unlabelled pixels, and a verdict.
Outputs in --out-dir: inspect_instances.csv, inspect_summary.txt, png/<target>.png (rows = instances where
untrained SAM succeeds and the 11-dataset decoder fails, then random ones; columns = systems).
"""

import argparse
import csv
import json
import os
import sys
import textwrap
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decoder_lib as dl  # noqa: E402
import emb_common as ec  # noqa: E402

TARGETS = [("zero-shot", "MSK_shoulder", 180.0, "scapula cartilage"),
           ("zero-shot", "MSK_shoulder", 225.0, "humerus cartilage"),
           ("knee", "MSK_FLASH", 2.0, "femoral cartilage"),
           ("knee", "MSK_FLASH", 4.0, "tibial cartilage")]
KNEE_NAMES = {1.0: "femur", 2.0: "femoral cartilage", 3.0: "tibia", 4.0: "tibial cartilage", 5.0: "patella",
              6.0: "patellar cartilage", 180.0: "scapula cartilage", 225.0: "humerus cartilage"}


def verdict(rec):
    if rec["pred_area"] < max(5, 0.02 * rec["gt_area"]):
        return "empty"
    if rec["dice"] >= 0.5:
        return "on target"
    if rec["share_on_other"] >= 0.5:
        return f"other structure ({rec['main_other']})"
    if rec["share_unlabelled"] >= 0.5:
        return "unlabelled tissue" + (" filling the box" if rec["box_fill"] >= 0.5 else "")
    return "partly on target"


def predict(model, wd, rows_items, device):
    """rows_items: row -> [(label, mask)]; yields (row, label, gt mask, predicted mask, native box, dice)."""
    for r, items in rows_items.items():
        geom = wd.geom(r)
        emb = torch.as_tensor(wd.emb(r), dtype=torch.float32, device=device)[None]
        nboxes = [ec.box_from_mask(m) for _, m in items]
        boxes = torch.as_tensor(np.stack([ec.box_to_input(b, geom) for b in nboxes]), dtype=torch.float32, device=device)
        with torch.no_grad():
            low = model(emb, boxes)[:, 0].float().cpu().numpy()
        for (lab, m), lg, b in zip(items, low, nboxes):
            p = ec.grid_to_native(lg, geom) > 0
            yield r, lab, m, p, b, ec.dice(p, m)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zeroshot-dir", required=True)
    ap.add_argument("--knee-dir", required=True)
    ap.add_argument("--swap-dir", required=True, help="decoder_swap/ (holds runs, runs_curve, runs_lora, results_knee)")
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-png", type=int, default=8, help="instances per target in the overlay figure")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    t0 = time.time()
    device = torch.device(args.device)
    os.makedirs(os.path.join(args.out_dir, "png"), exist_ok=True)
    print(sys.executable, "| torch", torch.__version__, "| device", device)

    S = args.swap_dir
    systems = [("SAM untrained", None),
               ("decoder trained, small data", os.path.join(S, "runs", "sam-enc_sam-dec_small", "seed0")),
               ("decoder LoRA, small data", os.path.join(S, "runs", "sam-enc_sam-dec-lora_small", "seed0")),
               ("decoder trained, 500 per dataset", os.path.join(S, "runs_curve", "sam-enc_sam-dec_n500", "seed0")),
               ("decoder trained, 500 per dataset + MSK_T2",
                os.path.join(S, "runs_curve", "sam-enc_sam-dec_n500_plus-MSK_T2", "seed0")),
               ("LoRA on encoder and decoder, small data", os.path.join(S, "runs_lora", "sam-enc-lora_sam-dec-lora_small", "seed0"))]
    dirs = {"zero-shot": args.zeroshot_dir, "knee": args.knee_dir}

    # instances of the targets (all labels of those slices are kept for the overlap shares)
    frozen = {t: dl.WorkDir(d, "sam_vit_b__sam") for t, d in dirs.items()}
    want = defaultdict(lambda: defaultdict(list))  # test set -> row -> [(label, mask)]
    segs = {}
    for t, wd in frozen.items():
        dsets = {ds for tt, ds, _, _ in TARGETS if tt == t}
        labs = {lab for tt, _, lab, _ in TARGETS if tt == t}
        for r in wd.rows("test"):
            m = wd.manifest[r]
            if m["dataset"] not in dsets:
                continue
            seg = ec.load_slice(m["img_path"], m["seg_path"])[1]
            segs[(t, r)] = seg
            for lab in ec.slice_labels(seg):
                if float(lab) in labs:
                    want[t][r].append((float(lab), seg == lab))
    for t in want:
        print(f"{t}: {sum(len(v) for v in want[t].values())} target structures in {len(want[t])} slices")

    recs, masks = [], {}
    for name, run in systems:
        ts = time.time()
        if run is None:
            model, wds = dl.build_decoder("sam", args.ckpt_dir, device, log=lambda *a: None).eval(), frozen
        elif os.path.basename(os.path.dirname(run)).startswith("sam-enc-lora"):
            import train_lora as tl
            meta = json.load(open(os.path.join(run, "meta.json")))
            _, encoder = ec.load_model("sam_vit_b", args.ckpt_dir, device)
            model = dl.build_decoder("sam", args.ckpt_dir, device, log=lambda *a: None)
            tl.add_lora(encoder, meta.get("rank", 4), meta.get("alpha", 4.0), False)
            dl.add_decoder_lora(model.mask_decoder, meta.get("rank", 4), meta.get("alpha", 4.0))
            st = torch.load(os.path.join(run, "best.pt"), map_location="cpu")
            tl.load_lora_state(encoder, st["lora"])
            model.mask_decoder.load_state_dict(st["decoder"])
            encoder.eval(), model.eval()
            wds = {t: tl.LiveWorkDir(d, encoder, device, device.type == "cuda") for t, d in dirs.items()}
        else:
            if not os.path.isfile(os.path.join(run, "best_decoder.pt")):
                print(f"  WARNING: {run}/best_decoder.pt missing - {name} skipped")
                continue
            model, _ = dl.load_trained_decoder(run, args.ckpt_dir, device)
            wds = frozen
        for t, rows_items in want.items():
            for r, lab, m, p, b, d in predict(model, wds[t], rows_items, device):
                seg = segs[(t, r)]
                x0, y0, x1, y1 = (int(v) for v in b)
                in_box = np.zeros_like(m)
                in_box[y0:y1 + 1, x0:x1 + 1] = True
                pa = int(p.sum())
                other = {float(o): float((p & (seg == o)).sum()) / max(pa, 1) for o in ec.slice_labels(seg) if o != lab}
                main_other = max(other, key=other.get) if other else None
                rec = {"system": name, "test_set": t, "dataset": wds[t].manifest[r]["dataset"], "row": r,
                       "label": lab, "structure": KNEE_NAMES.get(lab, str(lab)), "dice": d, "gt_area": int(m.sum()),
                       "pred_area": pa, "share_in_box": float((p & in_box).sum()) / max(pa, 1),
                       "box_fill": float((p & in_box).sum()) / max(int(in_box.sum()), 1),
                       "share_on_target": float((p & m).sum()) / max(pa, 1),
                       "share_on_other": float(sum(other.values())), "main_other": KNEE_NAMES.get(main_other, main_other),
                       "share_unlabelled": float((p & (seg == 0)).sum()) / max(pa, 1)}
                rec["verdict"] = verdict(rec)
                recs.append(rec)
                masks[(name, t, r, lab)] = (p, b)
        print(f"  {name}: done ({time.time() - ts:.0f} s)", flush=True)
        del model

    with open(os.path.join(args.out_dir, "inspect_instances.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)

    # ---- check against the evaluation: the same Dice must come out --------------------------------------
    lines = []
    ev = os.path.join(S, "results_knee", "eval_instances.csv")
    sys_id = {"SAM untrained": "ref_sam-original", "decoder trained, small data": "sam-enc_sam-dec_small",
              "decoder LoRA, small data": "sam-enc_sam-dec-lora_small", "decoder trained, 500 per dataset": "sam-enc_sam-dec_n500",
              "decoder trained, 500 per dataset + MSK_T2": "sam-enc_sam-dec_n500_plus-MSK_T2",
              "LoRA on encoder and decoder, small data": "sam-enc-lora_sam-dec-lora_small"}
    if os.path.isfile(ev):
        keys = {(sys_id[r["system"]], r["test_set"], int(r["row"]), float(r["label"])): r["dice"] for r in recs}
        diffs = []
        with open(ev) as f:
            for e in csv.DictReader(f):
                k = (e["system"], e["test_set"], int(e["row"]), float(e["label"]))
                if k in keys and int(e["seed"]) == 0:
                    diffs.append(abs(keys[k] - float(e["dice"])))
        lines.append(f"CHECK against {ev}: {len(diffs)} instances matched, max |Dice difference| "
                     f"{max(diffs) if diffs else float('nan'):.4f} (should be ~0; the encoder-LoRA arm may differ "
                     f"slightly from bf16)")
        lines.append("")

    # ---- summary ------------------------------------------------------------------------------------------
    lines.append("WHAT EACH SYSTEM PREDICTS (median over structures; verdict shares in %)")
    for t, ds, lab, sname in TARGETS:
        lines.append(f"\n  {ds} {sname} (label {lab:g})")
        for name, _ in systems:
            v = [r for r in recs if r["system"] == name and r["dataset"] == ds and r["label"] == lab]
            if not v:
                continue
            cnt = defaultdict(int)
            for r in v:
                cnt[r["verdict"]] += 1
            med = lambda k: float(np.median([r[k] for r in v]))  # noqa: E731
            lines.append(f"    {name:44s} Dice {med('dice'):.3f} | pred/GT area {med('pred_area') / max(med('gt_area'), 1):.2f}"
                         f" | in box {med('share_in_box'):.2f} | box filled {med('box_fill'):.2f} | "
                         + ", ".join(f"{k} {100 * c / len(v):.0f}" for k, c in sorted(cnt.items(), key=lambda x: -x[1])))
    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(args.out_dir, "inspect_summary.txt"), "w") as f:
        f.write(report + "\n")

    # ---- overlays -----------------------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [n for n, _ in systems if any(r["system"] == n for r in recs)]
    rng = np.random.default_rng(0)
    for t, ds, lab, sname in TARGETS:
        by = {(r["system"], r["row"]): r for r in recs if r["dataset"] == ds and r["label"] == lab}
        rows = sorted({r for (_, r) in by})
        if not rows:
            continue
        fail = [r for r in rows if by.get(("SAM untrained", r), {}).get("dice", 0) > 0.5
                and by.get(("decoder trained, small data", r), {}).get("dice", 1) < 0.05]
        pick = list(rng.permutation(fail)[:args.n_png - 2]) if fail else []
        rest = [r for r in rows if r not in pick]
        pick += list(rng.permutation(rest)[:args.n_png - len(pick)])
        fig, axes = plt.subplots(len(pick), len(names), figsize=(2.6 * len(names), 2.6 * len(pick)), squeeze=False)
        wd = frozen[t]
        for i, r in enumerate(pick):
            raw = ec.load_slice(wd.manifest[r]["img_path"])
            gt = segs[(t, r)] == lab
            x0, y0, x1, y1 = (int(v) for v in ec.box_from_mask(gt))
            pad = max(20, (x1 - x0) // 2, (y1 - y0) // 2)
            ya, yb, xa, xb = max(0, y0 - pad), min(raw.shape[0], y1 + pad + 1), max(0, x0 - pad), min(raw.shape[1], x1 + pad + 1)
            for j, name in enumerate(names):
                ax = axes[i, j]
                ax.imshow(raw[ya:yb, xa:xb], cmap="gray")
                ax.contour(gt[ya:yb, xa:xb], levels=[0.5], colors="lime", linewidths=0.8)
                if (name, t, r, lab) in masks:
                    p = masks[(name, t, r, lab)][0]
                    if p[ya:yb, xa:xb].any():
                        ax.contour(p[ya:yb, xa:xb], levels=[0.5], colors="red", linewidths=0.8)
                ax.add_patch(plt.Rectangle((x0 - xa, y0 - ya), x1 - x0, y1 - y0, fill=False, ec="yellow", lw=0.6, ls="--"))
                rec = by.get((name, r))
                title = "\n".join(textwrap.wrap(name, 24))
                ax.set_title(f"{title}\nDice {rec['dice']:.2f}" if rec else title, fontsize=6)
                ax.set_xticks([]), ax.set_yticks([])
                if j == 0:
                    ax.set_ylabel(f"row {r}", fontsize=6)
        fig.suptitle(f"{ds} {sname}: green = ground truth, red = prediction, yellow = prompt box", fontsize=8)
        fig.tight_layout()
        out = os.path.join(args.out_dir, "png", f"{ds}_{sname.replace(' ', '_')}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  figure: {out}")
    print(f"done ({(time.time() - t0) / 60:.1f} min) -> {args.out_dir}")


if __name__ == "__main__":
    main()
