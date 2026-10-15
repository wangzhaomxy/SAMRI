# -*- coding: utf-8 -*-
"""
Shared pieces for the swapped-encoder decoder experiments (train_decoder.py,
eval_decoder.py).

Question: does the encoder matter for SAMRI's decoder? A SAM mask decoder is
trained, with SAMRI's recipe, on the frozen output of either SAM's ViT-B
encoder (SAM preprocessing) or MedSAM's encoder (MedSAM preprocessing). Only the
encoder differs between arms; data, decoder start, recipe and seeds are shared.

Targets and prompts follow SAMRI's training exactly (train_multi_gpus.py):
  * the native mask is resized to the 256x256 low-res frame of the decoder
    output with nearest neighbour: 'sam' = longer side to 256 + zero pad
    (utils.utils.preprocess_mask), 'medsam' = stretch to 256x256;
  * the box is taken from that 256-frame mask with +-jitter px (utils.utils.
    gen_bboxes: only applied when the side is longer than jitter + 10) and
    scaled x4 into the 1024 input frame;
  * loss = DiceFocalLoss(sigmoid, squared_pred, lambda_dice=1, lambda_focal=10);
  * only the mask decoder trains; SAM's prompt encoder is frozen and shared.
Testing follows SAMRI's test: a tight box from the native mask (no jitter)
mapped into the 1024 frame, one mask, prediction mapped back to native pixels.
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emb_common as ec  # noqa: E402

LOWRES = 256
SAM_CKPT, MEDSAM_CKPT = "sam_vit_b_01ec64.pth", "medsam_vit_b.pth"

# arm id -> (features condition, decoder start, plain-language name)
ARMS = {
    "sam-enc_sam-dec_small": ("sam_vit_b__sam", "sam", "SAM encoder, SAM decoder start, small data"),
    "medsam-enc_sam-dec_small": ("medsam_vit_b__medsam", "sam", "MedSAM encoder, SAM decoder start, small data"),
    "medsam-enc_medsam-dec_small": ("medsam_vit_b__medsam", "medsam",
                                    "MedSAM encoder, MedSAM decoder start, small data"),
    "sam-enc_sam-dec_all11": ("sam_vit_b__sam", "sam",
                              "SAM encoder, SAM decoder start, all training slices of the 11 datasets"),
}


# ---------------------------------------------------------------------------
# targets and prompts
# ---------------------------------------------------------------------------
def lowres_target(mask, pipeline):
    """Native binary mask (H, W) -> (256, 256) uint8 in the decoder-output frame."""
    H, W = mask.shape
    m = torch.from_numpy(mask.astype(np.float32))[None, None]
    if pipeline == "sam":  # utils.utils.preprocess_mask
        s = LOWRES * 1.0 / max(H, W)
        h, w = int(H * s + 0.5), int(W * s + 0.5)
        m = F.interpolate(m, (h, w), mode="nearest")
        m = F.pad(m, (0, LOWRES - w, 0, LOWRES - h))
    else:
        m = F.interpolate(m, (LOWRES, LOWRES), mode="nearest")
    return (m[0, 0] > 0.5).numpy().astype(np.uint8)


def tight_box_256(t):
    ys, xs = np.nonzero(t)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], np.int64)


def jitter_box_1024(box256, jitter, rng):
    """utils.utils.gen_bboxes on the 256-frame mask, then x4 (apply_boxes_torch
    with original_size=(256, 256))."""
    x0, y0, x1, y1 = [int(v) for v in box256]
    b = [x0, y0, x1, y1]
    if jitter:
        if y1 - y0 > jitter + 10:
            b[1] = max(0, y0 + int(rng.integers(-jitter, jitter + 1)))
            b[3] = min(LOWRES, y1 + int(rng.integers(-jitter, jitter + 1)))
        if x1 - x0 > jitter + 10:
            b[0] = max(0, x0 + int(rng.integers(-jitter, jitter + 1)))
            b[2] = min(LOWRES, x1 + int(rng.integers(-jitter, jitter + 1)))
    return np.array(b, np.float32) * (ec.IMG_SIZE / LOWRES)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
class PromptedDecoder(nn.Module):
    """SAM prompt encoder (frozen) + mask decoder (trainable)."""

    def __init__(self, prompt_encoder, mask_decoder):
        super().__init__()
        self.prompt_encoder, self.mask_decoder = prompt_encoder, mask_decoder
        for p in self.prompt_encoder.parameters():
            p.requires_grad = False

    def forward(self, emb, boxes):
        """emb (B, 256, 64, 64) with boxes (B, 4), or (1, ...) with boxes (N, 4)."""
        sparse, dense = self.prompt_encoder(points=None, boxes=boxes, masks=None)
        low, _ = self.mask_decoder(image_embeddings=emb, image_pe=self.prompt_encoder.get_dense_pe(),
                                   sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                                   multimask_output=False)
        return low  # (B or N, 1, 256, 256) logits


def _load_sam(path):
    from segment_anything import sam_model_registry
    if not os.path.isfile(path):  # build_sam would otherwise prompt for a download
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return sam_model_registry["vit_b"](checkpoint=path).eval()


def build_decoder(decoder_start, ckpt_dir, device, log=print):
    """decoder_start: 'sam' | 'medsam' | path to any ViT-B SAM-format checkpoint
    (e.g. a released SAMRI .pth). The prompt encoder is always SAM's original."""
    sam = _load_sam(os.path.join(ckpt_dir, SAM_CKPT))
    pe = sam.prompt_encoder
    if decoder_start == "sam":
        md = sam.mask_decoder
    else:
        path = os.path.join(ckpt_dir, MEDSAM_CKPT) if decoder_start == "medsam" else decoder_start
        other = _load_sam(path)
        md = other.mask_decoder
        diff = max((a - b).abs().max().item() for a, b in
                   zip(pe.state_dict().values(), other.prompt_encoder.state_dict().values()))
        log(f"  prompt encoder of {os.path.basename(path)} vs SAM's: max |difference| {diff:.2e} "
            f"({'identical' if diff < 1e-6 else 'DIFFERENT - SAM prompt encoder is used'})")
        del other
    del sam.image_encoder
    return PromptedDecoder(pe, md).to(device)


def load_trained_decoder(run_dir, ckpt_dir, device):
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    model = build_decoder("sam", ckpt_dir, device, log=lambda *a: None)
    model.mask_decoder.load_state_dict(torch.load(os.path.join(run_dir, "best_decoder.pt"), map_location=device))
    return model.eval(), meta


# ---------------------------------------------------------------------------
# data from a work dir (manifest + geometry + features)
# ---------------------------------------------------------------------------
class WorkDir:
    def __init__(self, work_dir, cond):
        self.dir = work_dir
        self.manifest = ec.read_manifest(os.path.join(work_dir, "manifest.csv"))
        self.cond = cond
        self.pipeline = ec.parse_cond(cond)[1]
        fdir = os.path.join(work_dir, "features", cond)
        if not os.path.isfile(os.path.join(fdir, "DONE")):
            raise FileNotFoundError(f"features not finished: {fdir}/DONE missing")
        self.neck = np.load(os.path.join(fdir, "neck.npy"), mmap_mode="r")
        self.index = {int(r): i for i, r in enumerate(np.load(os.path.join(fdir, "neck_rows.npy")))}

    def rows(self, split):
        return [r["row"] for r in self.manifest if r["split"] == split and r["row"] in self.index]

    def emb(self, r):
        return np.array(self.neck[self.index[r]])  # copy: the memmap is read-only

    def instances(self, rows):
        """[(row, label, native mask)] for the given rows (reads the label maps)."""
        out = []
        for r in rows:
            seg = ec.load_slice(self.manifest[r]["img_path"], self.manifest[r]["seg_path"])[1]
            for lab in ec.slice_labels(seg):
                out.append((r, float(lab), seg == lab))
        return out

    def geom(self, r):
        m = self.manifest[r]
        return ec.make_geom(self.pipeline, m["H"], m["W"])


def val_split(manifest, frac, seed, path):
    """Held-out training rows per dataset for choosing the epoch; saved once and
    reused by every arm and seed so all arms see the same validation slices."""
    if os.path.isfile(path):
        return json.load(open(path))
    rng = np.random.default_rng(seed)
    by_ds = {}
    for r in manifest:
        if r["split"] == "train":
            by_ds.setdefault(r["dataset"], []).append(r["row"])
    val = []
    for ds in sorted(by_ds):
        rows = sorted(by_ds[ds])
        k = max(1, int(round(frac * len(rows))))
        val += sorted(int(x) for x in rng.choice(rows, k, replace=False))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(val, f)
    return val


@torch.no_grad()
def predict_instances(model, wd, instances, device):
    """Tight native box per instance -> native binary prediction; yields
    (row, label, dice, area). One encoder output per slice, all its boxes batched."""
    by_row = {}
    for r, lab, m in instances:
        by_row.setdefault(r, []).append((lab, m))
    for r, items in by_row.items():
        geom = wd.geom(r)
        emb = torch.as_tensor(wd.emb(r), dtype=torch.float32, device=device)[None]
        boxes = torch.as_tensor(np.stack([ec.box_to_input(ec.box_from_mask(m), geom) for _, m in items]),
                                dtype=torch.float32, device=device)
        low = model(emb, boxes)[:, 0].float().cpu().numpy()
        for (lab, m), lg in zip(items, low):
            yield r, lab, ec.dice(ec.grid_to_native(lg, geom) > 0, m), int(m.sum())
