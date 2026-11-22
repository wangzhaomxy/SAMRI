#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Typical test slices for the qualitative figure of the SAMRI paper.

Runs the four models of the figure on the structures listed in typical_samples.csv, compares every Dice with
the Dice in the benchmark records, and saves the masks and preview pictures. Nothing is trained.

    model               prompt                      predicted as in
    SAM ViT-B           tight box                   evaluation/test_vis.py  (utils/visual.get_test_record_from_ds)
    SAMRI box           tight box                   evaluation/test_vis.py
    SAMRI box+point     tight box + one point       evaluation/test_vis.py
    MedSAM              tight box                   evaluation/test_medsam.py  (MedSAM's own MR preprocessing)

The point of SAMRI box+point is drawn by the paper's rule (utils.utils.gen_points) with a fixed seed per
structure. The points of the benchmark run were not seeded, so its Dice can differ a little from the record;
the three box-only models are deterministic and must reproduce the records.

Outputs in --out-dir
    dice_check.csv      one row per structure and model: Dice of this run, Dice in the records, difference
    npz/<panel>_c<k>.npz  image, reference mask, the four predicted masks, box, point (for the final figure)
    png/<panel>.png     every candidate of the panel: whole slice with the prompt box, and the zoom with all contours
    png/contact_sheet.png  candidate 1 of every panel

Second use, only if the SAMRI checkpoints do not reproduce the records: find the checkpoint that does.
    --scan-ckpts "/path/to/Model_save/*.pth" --scan-prompt box        (or box+point)
scores every checkpoint of the pattern on candidate 1 of each panel and ranks them by how close their Dice is
to the records. No pictures are made in this mode.

Run from anywhere; the repository root (the folder that holds segment_anything/ and utils/) is taken from
--repo-root, by default two folders above this file (evaluation/typical_samples/ -> repository root).
"""

import argparse
import csv
import glob
import json
import os
import random
import sys
import textwrap
import time

import numpy as np

MODEL_ORDER = ["sam_box", "samri_box", "samri_bp", "medsam"]
MODEL_NAME = {"sam_box": "SAM ViT-B (box)", "samri_box": "SAMRI box (box)",
              "samri_bp": "SAMRI box+point (box + point)", "medsam": "MedSAM (box)"}
# contour colours: same hues as the size figure of the paper (checked for colour-blind separation on a dark surface)
COLOUR = {"sam_box": "#3D9BE0", "medsam": "#E0559F", "samri_box": "#D55E00", "samri_bp": "#12A87E"}
LEGEND = {"sam_box": "SAM ViT-B", "medsam": "MedSAM", "samri_box": "SAMRI box", "samri_bp": "SAMRI box+point"}
TOL = 0.005          # a box-only Dice within this distance of the record counts as reproduced
TOL_POINT = 0.05     # box+point: the point is random, so a wider distance is expected


# --------------------------------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------------------------------
def read_samples(path):
    rows = list(csv.DictReader(open(path, newline="")))
    need = {"panel", "candidate", "description", "test_set", "dataset", "image", "label_value",
            "rec_sam_box", "rec_medsam", "rec_samri_box", "rec_samri_bp"}
    missing = need - set(rows[0])
    if missing:
        sys.exit(f"{path}: columns missing: {sorted(missing)}")
    for r in rows:
        r["candidate"] = int(r["candidate"])
        r["label_value"] = float(r["label_value"])
        r["pixel_count"] = int(float(r.get("pixel_count") or 0))
        for k in ("rec_sam_box", "rec_medsam", "rec_samri_box", "rec_samri_bp"):
            r[k] = float(r[k])
        r["distance"] = float(r["distance"]) if r.get("distance") not in (None, "") else None
        r["typical"] = int(r["typical"]) if r.get("typical") not in (None, "") else 1
        r["key"] = f"{r['panel']}_c{r['candidate']}"
    return rows


def resolve_files(rows, data_root):
    """Image and mask file of every row. The mask is the one the benchmark paired with the image:
    NiiDataset sorts the *_img_* and the *_seg_* files of a testing folder and pairs them by position."""
    cache, problems = {}, []
    for r in rows:
        folder = os.path.join(data_root, r["test_set"], r["dataset"], "testing", "")
        if folder not in cache:
            cache[folder] = (sorted(glob.glob(folder + "*_img_*")), sorted(glob.glob(folder + "*_seg_*")))
        imgs, segs = cache[folder]
        img = os.path.join(folder, r["image"])
        if not os.path.isdir(folder):
            problems.append(f"folder not found: {folder}")
            continue
        if img not in imgs:
            problems.append(f"image not found: {img}")
            continue
        if len(imgs) != len(segs):
            problems.append(f"{folder}: {len(imgs)} image files but {len(segs)} mask files")
            continue
        seg = segs[imgs.index(img)]
        by_name = "_seg_".join(img.rsplit("_img_", 1))
        if os.path.basename(seg) != os.path.basename(by_name):
            print(f"  note: {r['key']}: the benchmark pairs {os.path.basename(img)} with {os.path.basename(seg)} "
                  f"(by position), not with the mask of the same name")
        r["img_path"], r["seg_path"] = img, seg
    if problems:
        print("\n".join("  " + p for p in problems[:20]))
        sys.exit(f"{len(problems)} of {len(rows)} structures could not be found under {data_root} - nothing was run")


def load_slice(r):
    import nibabel as nib
    img = nib.load(r["img_path"]).get_fdata()[0]          # (H, W) raw intensities, as NiiDataset reads them
    seg = nib.load(r["seg_path"]).get_fdata()[0]          # (H, W) labels
    gt = seg == r["label_value"]
    if not gt.any():
        raise ValueError(f"label {r['label_value']} is not in {r['seg_path']} (labels: {np.unique(seg)[:12]})")
    return img, gt


def sam_input(img):
    """utils/dataloader.NiiDataset._preprocess: slice min-max to [0, 255], three channels, uint8."""
    x = np.array([img, img, img]).transpose(1, 2, 0)
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return np.round(x * 255).astype(np.uint8)


def dice(pred, gt, smooth=1e-10):
    """utils/losses.dice_similarity"""
    pred, gt = pred.astype(np.float64), gt.astype(np.float64)
    return float((2.0 * np.sum(pred * gt) + smooth) / (np.sum(pred) + np.sum(gt) + smooth))


# --------------------------------------------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------------------------------------------
def load_model(ckpt, device):
    import torch
    from segment_anything import sam_model_registry
    model = sam_model_registry["vit_b"](ckpt)             # same call as test_vis.py and test_medsam.py
    return model.to(device).eval()


def predict_sam(model, rows, slices, with_point, seed):
    """SAM / SAMRI through SamPredictor with a tight box (and one point), single mask output: test_vis.py."""
    from segment_anything import SamPredictor
    from utils.utils import gen_bboxes, gen_points
    predictor = SamPredictor(model)
    out = {}
    for r in rows:
        img, gt = slices[r["key"]]
        predictor.set_image(sam_input(img))
        box = gen_bboxes(gt, jitter=0)
        point = None
        if with_point:
            random.seed(f"{seed}/{r['key']}")             # one fixed point per structure, whatever the order
            point = gen_points(gt)
        masks, _, _ = predictor.predict(point_coords=point,
                                        point_labels=np.array([1]) if with_point else None,
                                        box=box[None, :], multimask_output=False)
        out[r["key"]] = (masks[0].astype(bool), np.asarray(box), None if point is None else np.asarray(point[0]))
    return out


def medsam_mr_preprocess(img_np):
    """Copied from evaluation/test_medsam.py (per-slice version of MedSAM-main/pre_CT_MR.py, MR branch)."""
    image_data = img_np.astype(np.float64)
    foreground = image_data[image_data > 0]
    if foreground.size == 0:
        return np.zeros_like(image_data, dtype=np.uint8)
    lower_bound, upper_bound = np.percentile(foreground, 0.5), np.percentile(foreground, 99.5)
    image_data_pre = np.clip(image_data, lower_bound, upper_bound)
    image_data_pre = ((image_data_pre - np.min(image_data_pre))
                      / max(np.max(image_data_pre) - np.min(image_data_pre), 1e-8) * 255.0)
    image_data_pre[image_data == 0] = 0
    return np.uint8(image_data_pre)


def predict_medsam(model, rows, slices, device):
    """MedSAM as in evaluation/test_medsam.py: its MR preprocessing, stretch to 1024 x 1024, box in that frame."""
    import torch
    import torch.nn.functional as F
    from skimage import transform
    out = {}
    for r in rows:
        img, gt = slices[r["key"]]
        img_np = medsam_mr_preprocess(img)
        img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
        H, W, _ = img_3c.shape
        img_1024 = transform.resize(img_3c, (1024, 1024), order=3, preserve_range=True,
                                    anti_aliasing=True).astype(np.uint8)
        img_1024 = (img_1024 - img_1024.min()) / np.clip(img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None)
        tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
        y, x = np.where(gt)
        box = np.array([x.min(), y.min(), x.max(), y.max()])
        box_1024 = np.array([box]) / np.array([W, H, W, H]) * 1024
        with torch.no_grad():
            emb = model.image_encoder(tensor)
            box_t = torch.as_tensor(box_1024, dtype=torch.float, device=device)[:, None, :]
            sparse, dense = model.prompt_encoder(points=None, boxes=box_t, masks=None)
            low, _ = model.mask_decoder(image_embeddings=emb, image_pe=model.prompt_encoder.get_dense_pe(),
                                        sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                                        multimask_output=False)
            prob = F.interpolate(torch.sigmoid(low), size=(H, W), mode="bilinear", align_corners=False)
        out[r["key"]] = ((prob.squeeze().cpu().numpy() > 0.5), box, None)
    return out


# --------------------------------------------------------------------------------------------------------------
# check against the records
# --------------------------------------------------------------------------------------------------------------
def report(model_key, rows, preds, slices, ckpt):
    rec_col = {"sam_box": "rec_sam_box", "samri_box": "rec_samri_box", "samri_bp": "rec_samri_bp",
               "medsam": "rec_medsam"}[model_key]
    tol = TOL_POINT if model_key == "samri_bp" else TOL
    print(f"\n===== {MODEL_NAME[model_key]} =====   checkpoint: {ckpt}")
    print(f"{'structure':34s} {'this run':>8s} {'record':>7s} {'diff':>7s}")
    lines, diffs = [], []
    for r in rows:
        d = dice(preds[r["key"]][0], slices[r["key"]][1])
        diff = d - r[rec_col]
        diffs.append(abs(diff))
        flag = "" if abs(diff) <= tol else "   <-- differs"
        print(f"{r['key']:34s} {d:8.4f} {r[rec_col]:7.4f} {diff:+7.4f}{flag}")
        lines.append(dict(structure=r["key"], model=model_key, dice=round(d, 4), record=r[rec_col],
                          difference=round(diff, 4), within_tolerance=int(abs(diff) <= tol)))
    n_ok = int(sum(x <= tol for x in diffs))
    share = n_ok / len(diffs)
    if model_key == "samri_bp":
        ok = np.median(diffs) <= 0.02
        verdict = ("agrees with the records (the point is random, so small differences are expected)" if ok
                   else "DOES NOT agree with the records")
    else:
        ok = share >= 0.9
        verdict = "REPRODUCES the records" if ok else "DOES NOT reproduce the records"
    print(f"{MODEL_NAME[model_key]}: {n_ok} of {len(diffs)} structures within {tol} of the records, "
          f"median difference {np.median(diffs):.4f}, largest {max(diffs):.4f}  ->  {verdict}")
    if not ok and model_key == "sam_box":
        print("!!!!! SAM ViT-B uses the public weights, so this means the data or the code differ from the "
              "benchmark run. Stop here and check before using anything from this job.")
    if not ok and model_key.startswith("samri"):
        print("!!!!! These are not the weights behind the paper's tables. Do not use the pictures of this job;\n"
              "      find the right checkpoint first (see --scan-ckpts in the header of this file).")
    return lines, ok


# --------------------------------------------------------------------------------------------------------------
# pictures
# --------------------------------------------------------------------------------------------------------------
def zoom_window(box, H, W, margin=0.55, min_side=36):
    x0, y0, x1, y1 = [float(v) for v in box]
    side = max(max(x1 - x0, y1 - y0) * (1 + 2 * margin), min_side)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    xa, xb = int(max(0, np.floor(cx - side / 2))), int(min(W, np.ceil(cx + side / 2)))
    ya, yb = int(max(0, np.floor(cy - side / 2))), int(min(H, np.ceil(cy + side / 2)))
    return xa, xb, ya, yb


def draw_full(ax, item):
    """Whole slice with the prompt box (cyan) and the frame of the zoom (yellow, dashed)."""
    from matplotlib.patches import Rectangle
    H, W = item["gt"].shape
    xa, xb, ya, yb = zoom_window(item["box"], H, W)
    ax.imshow(item["image"], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    x0, y0, x1, y1 = [int(v) for v in item["box"]]
    ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0 + 1, y1 - y0 + 1, fill=False, ec="#35D0E0", lw=1.0))
    ax.add_patch(Rectangle((xa - 0.5, ya - 0.5), xb - xa, yb - ya, fill=False, ec="#F2D200", lw=1.0, ls="--"))
    ax.set_xticks([]); ax.set_yticks([])


def draw_zoom(ax, item):
    """Zoom on the structure: reference mask filled, one contour per model, the point of box+point."""
    img, gt = item["image"], item["gt"]
    H, W = gt.shape
    xa, xb, ya, yb = zoom_window(item["box"], H, W)
    extent = (xa - 0.5, xb - 0.5, yb - 0.5, ya - 0.5)
    ax.imshow(img[ya:yb, xa:xb], cmap="gray", vmin=0, vmax=255, interpolation="nearest", extent=extent)
    fill = np.zeros((yb - ya, xb - xa, 4))
    fill[gt[ya:yb, xa:xb]] = (1.0, 0.83, 0.0, 0.42)
    ax.imshow(fill, interpolation="nearest", extent=extent)
    xs, ys = np.arange(xa, xb), np.arange(ya, yb)
    for k in ("medsam", "sam_box", "samri_box", "samri_bp"):
        m = item[k][ya:yb, xa:xb]
        if m.any() and not m.all() and m.shape[0] > 1 and m.shape[1] > 1:
            ax.contour(xs, ys, m.astype(float), levels=[0.5], colors=[COLOUR[k]], linewidths=1.5)
    if item.get("point") is not None:
        ax.plot([item["point"][0]], [item["point"][1]], marker="+", ms=7, mew=1.4, color=COLOUR["samri_bp"])
    ax.set_xlim(xa - 0.5, xb - 0.5)
    ax.set_ylim(yb - 0.5, ya - 0.5)
    ax.set_xticks([]); ax.set_yticks([])


def legend_handles():
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = [Patch(fc=(1.0, 0.83, 0.0, 0.6), ec="none", label="Reference mask")]
    h += [Line2D([0], [0], color=COLOUR[k], lw=2, label=LEGEND[k]) for k in ("sam_box", "medsam", "samri_box", "samri_bp")]
    h += [Line2D([0], [0], color="#35D0E0", lw=1.2, label="Prompt box"),
          Line2D([0], [0], color=COLOUR["samri_bp"], lw=0, marker="+", ms=8, mew=1.4, label="Point (box+point only)")]
    return h


def make_pictures(rows, items, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.join(out_dir, "png"), exist_ok=True)
    panels = []
    for r in rows:
        if r["panel"] not in panels:
            panels.append(r["panel"])
    per_row = 2                                               # candidates per row: [slice, zoom] x 2
    for p in panels:
        rs = [r for r in rows if r["panel"] == p]
        nrow = int(np.ceil(len(rs) / per_row))
        fig, axes = plt.subplots(nrow, 2 * per_row, figsize=(6.6 * per_row, 3.45 * nrow + 1.0), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for j, r in enumerate(rs):
            ax_f, ax_z = axes[j // per_row][2 * (j % per_row)], axes[j // per_row][2 * (j % per_row) + 1]
            ax_f.axis("on"); ax_z.axis("on")
            it = items[r["key"]]
            draw_full(ax_f, it)
            draw_zoom(ax_z, it)
            note = "" if r["distance"] is None else (f"   distance from the medians {r['distance']:.3f}"
                                                     + ("" if r["typical"] else "  (NOT typical)"))
            ax_f.set_title(f"candidate {r['candidate']}{note}\n{r['image'][-50:]}", fontsize=6.5, loc="left")
            d = it["dice"]
            ax_z.set_title(f"Dice:  SAM ViT-B {d['sam_box']:.2f}   MedSAM {d['medsam']:.2f}\n"
                           f"SAMRI box {d['samri_box']:.2f}   SAMRI box+point {d['samri_bp']:.2f}",
                           fontsize=7, loc="left")
        fig.suptitle(f"{p}   {rs[0]['description']}", fontsize=10, x=0.01, ha="left")
        fig.legend(handles=legend_handles(), loc="lower center", ncol=7, fontsize=7, frameon=False)
        fig.tight_layout(rect=(0, 0.35 / (3.45 * nrow + 1.0), 1, 1 - 0.3 / (3.45 * nrow + 1.0)), h_pad=1.8)
        fig.savefig(os.path.join(out_dir, "png", p + ".png"), dpi=120)
        plt.close(fig)
    first = [r for r in rows if r["candidate"] == 1]
    ncol = min(6, len(first))
    nrow = int(np.ceil(len(first) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.75 * nrow + 0.7), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, r in zip(axes.ravel(), first):
        ax.axis("on")
        draw_zoom(ax, items[r["key"]])
        ax.set_title("\n".join(textwrap.wrap(r["description"], 36)), fontsize=7)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=7, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.5 / (2.75 * nrow + 0.7), 1, 1))
    fig.savefig(os.path.join(out_dir, "png", "contact_sheet.png"), dpi=130)
    plt.close(fig)
    return len(panels)


# --------------------------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------------------------
def scan(args, rows, slices, device):
    import torch
    with_point = args.scan_prompt == "box+point"
    rec_col = "rec_samri_bp" if with_point else "rec_samri_box"
    files = sorted(glob.glob(args.scan_ckpts))
    if not files:
        sys.exit(f"no file matches {args.scan_ckpts}")
    rows = [r for r in rows if r["candidate"] == 1]
    print(f"\nscanning {len(files)} checkpoints with the {args.scan_prompt} prompt on {len(rows)} structures "
          f"(candidate 1 of each panel); record column: {rec_col}")
    print(f"{'checkpoint':44s} {'mean Dice':>9s} {'mean |diff|':>11s} {'within ' + str(TOL):>13s}")
    res = []
    for f in files:
        try:
            model = load_model(f, device)
        except Exception as e:                            # a file that is not a full SAM state dict
            print(f"{os.path.basename(f):44s} could not be loaded: {str(e)[:60]}")
            continue
        preds = predict_sam(model, rows, slices, with_point, args.seed)
        d = np.array([dice(preds[r["key"]][0], slices[r["key"]][1]) for r in rows])
        diff = np.abs(d - np.array([r[rec_col] for r in rows]))
        res.append((float(diff.mean()), os.path.basename(f), float(d.mean()), int((diff <= TOL).sum())))
        print(f"{os.path.basename(f):44s} {d.mean():9.4f} {diff.mean():11.4f} {int((diff <= TOL).sum()):6d} of {len(rows)}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    res.sort()
    print("\nclosest to the records:")
    for mean_diff, name, mean_d, n in res[:5]:
        print(f"  {name:44s} mean |diff| {mean_diff:.4f}, {n} of {len(rows)} within {TOL}")
    if res and with_point:
        print("(box+point: the point is random, so expect a mean |diff| of about 0.01-0.03 for the right checkpoint)")
    elif res and res[0][0] > 0.01:
        print("none of these checkpoints reproduces the box records (the right one gives a mean |diff| near 0)")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", default=os.path.join(here, "typical_samples.csv"))
    ap.add_argument("--data-root", required=True, help="folder that holds SAMRI_train_test/ and Zeroshot/")
    ap.add_argument("--sam-ckpt", required=True)
    ap.add_argument("--medsam-ckpt", required=True)
    ap.add_argument("--samri-box-ckpt", required=True)
    ap.add_argument("--samri-bp-ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--repo-root", default=os.path.abspath(os.path.join(here, "..", "..")))
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if a GPU is visible)")
    ap.add_argument("--seed", type=int, default=0, help="seed of the point of SAMRI box+point")
    ap.add_argument("--limit", type=int, default=0, help="only the first N structures (quick test)")
    ap.add_argument("--no-pictures", action="store_true")
    ap.add_argument("--scan-ckpts", default="", help='pattern of SAMRI checkpoints to rank against the records, e.g. "/path/Model_save/*.pth"')
    ap.add_argument("--scan-prompt", default="box", choices=["box", "box+point"])
    args = ap.parse_args()
    t0 = time.time()

    if not os.path.isdir(os.path.join(args.repo_root, "segment_anything")):
        sys.exit(f"--repo-root {args.repo_root} has no segment_anything/ folder; pass the SAMRI repository root")
    sys.path.insert(0, args.repo_root)
    import torch
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(sys.executable, "| torch", torch.__version__, "| device:",
          torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")
    print("repository root:", args.repo_root)

    ckpts = {"sam_box": args.sam_ckpt, "samri_box": args.samri_box_ckpt, "samri_bp": args.samri_bp_ckpt,
             "medsam": args.medsam_ckpt}
    missing = [f"{k}: {v}" for k, v in ckpts.items() if not os.path.isfile(v)]
    if missing and not args.scan_ckpts:
        sys.exit("checkpoint file(s) not found:\n  " + "\n  ".join(missing))

    rows = read_samples(args.samples)
    if args.limit:
        rows = rows[:args.limit]
    resolve_files(rows, args.data_root)
    slices = {}
    for r in rows:
        img, gt = load_slice(r)
        slices[r["key"]] = (img, gt)
        if r["pixel_count"] and int(gt.sum()) != r["pixel_count"]:
            print(f"  note: {r['key']}: the structure has {int(gt.sum())} pixels, the records say {r['pixel_count']}")
    print(f"{len(rows)} structures listed in {os.path.basename(args.samples)}; all image and mask files found and read")

    if args.scan_ckpts:
        scan(args, rows, slices, device)
        print(f"\nfinished in {time.time() - t0:.0f} s")
        return 0

    os.makedirs(os.path.join(args.out_dir, "npz"), exist_ok=True)
    all_lines, all_ok, preds = [], {}, {}
    for key in MODEL_ORDER:
        ts = time.time()
        model = load_model(ckpts[key], device)
        if key == "medsam":
            preds[key] = predict_medsam(model, rows, slices, device)
        else:
            preds[key] = predict_sam(model, rows, slices, with_point=(key == "samri_bp"), seed=args.seed)
        lines, ok = report(key, rows, preds[key], slices, ckpts[key])
        all_lines += lines
        all_ok[key] = ok
        print(f"({time.time() - ts:.0f} s)", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with open(os.path.join(args.out_dir, "dice_check.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_lines[0]))
        w.writeheader(); w.writerows(all_lines)

    items = {}
    for r in rows:
        img, gt = slices[r["key"]]
        it = dict(image=sam_input(img)[..., 0], gt=gt, box=preds["sam_box"][r["key"]][1],
                  point=preds["samri_bp"][r["key"]][2],
                  dice={k: dice(preds[k][r["key"]][0], gt) for k in MODEL_ORDER})
        for k in MODEL_ORDER:
            it[k] = preds[k][r["key"]][0]
        items[r["key"]] = it
        meta = dict(r)                                    # the whole row of the list, with the file paths
        meta.update(dice=it["dice"], checkpoints={k: os.path.basename(v) for k, v in ckpts.items()}, seed=args.seed)
        np.savez_compressed(os.path.join(args.out_dir, "npz", r["key"] + ".npz"), image=it["image"], gt=gt,
                            sam_box=it["sam_box"], samri_box=it["samri_box"], samri_bp=it["samri_bp"],
                            medsam=it["medsam"], box=np.asarray(it["box"]),
                            point=np.asarray(it["point"] if it["point"] is not None else [-1, -1]),
                            meta=json.dumps(meta))
    print(f"\nmasks saved: {len(items)} files in {os.path.join(args.out_dir, 'npz')}")

    if not args.no_pictures:
        try:
            n = make_pictures(rows, items, args.out_dir)
            print(f"pictures saved: {n} panels and contact_sheet.png in {os.path.join(args.out_dir, 'png')}")
        except Exception as e:                            # the masks are already saved; pictures can be redrawn later
            print(f"!!!!! pictures failed ({type(e).__name__}: {e}); the npz files are complete")

    print("\n===== summary =====")
    for key in MODEL_ORDER:
        print(f"  {MODEL_NAME[key]:32s} {'ok' if all_ok[key] else 'DOES NOT MATCH THE RECORDS'}")
    print(f"finished in {time.time() - t0:.0f} s  ->  {args.out_dir}")
    return 0 if all(all_ok.values()) else 2


if __name__ == "__main__":
    sys.exit(main())
