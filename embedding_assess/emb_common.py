# -*- coding: utf-8 -*-
"""
Shared helpers for the encoder-embedding experiments in embedding_assess/.

The experiments ask whether SAM's image encoder needs MRI fine-tuning, by
comparing frozen encoders (SAM ViT-B, MedSAM's fine-tuned ViT-B, SAM ViT-H, a
randomly initialised ViT-B) on the same MRI slices:

  * CKA      - how far MedSAM's fine-tuning moved the encoder's
               representation away from SAM's, layer by layer (cka.py).
  * probes   - how well the frozen embedding separates the prompted structure
               from its surroundings, with a tiny head instead of the SAM
               decoder (probe.py).

Every encoder is run under BOTH preprocessing pipelines (the "2x2 design"), so
encoders can be compared on identical inputs:

  "sam"     SAMRI / SAM input: per-slice min-max to uint8 [0, 255], 3 channels,
            resize the LONGER side to 1024 (ResizeLongestSide, as in
            SamPredictor.set_image), ImageNet mean/std, zero-pad bottom/right
            to 1024x1024. Identical to how SAMRI's stored embeddings were made.
  "medsam"  MedSAM input: MR percentile clip 0.5-99.5 -> uint8 [0, 255]
            (MedSAM-main/pre_CT_MR.py, per slice), bicubic STRETCH to
            1024x1024, min-max to [0, 1], no ImageNet normalisation
            (MedSAM_Inference.py). Identical to evaluation/test_medsam.py after
            the 14 Oct 2026 fix.

Geometry: "sam" keeps the aspect ratio and pads; "medsam" stretches. Masks are
mapped to the 64x64 token grid with the same geometry as the image, and
predictions are mapped back to the native slice size the same way SAM / MedSAM
map their masks back, so every score is computed on the original pixels.
"""

import csv
import os
import sys

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# segment_anything (and through it torchvision) is imported only inside the
# functions that need it, so build_manifest.py runs with just torch + nibabel.

IMG_SIZE = 1024
PATCH = 16
GRID = IMG_SIZE // PATCH  # 64 tokens per side
SAM_PIXEL_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
SAM_PIXEL_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
PIPELINES = ("sam", "medsam")
IMAGE_KEY, MASK_KEY = "_img_", "_seg_"

# name -> (segment_anything registry key, checkpoint file name; None = random init)
ENCODERS = {
    "sam_vit_b": ("vit_b", "sam_vit_b_01ec64.pth"),
    "medsam_vit_b": ("vit_b", "medsam_vit_b.pth"),
    "sam_vit_h": ("vit_h", "sam_vit_h_4b8939.pth"),
    "random_vit_b": ("vit_b", None),
}


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_slice(img_path, seg_path=None):
    """Raw intensities (H, W) float64 and, if given, the label map (H, W)."""
    raw = np.asarray(nib.load(img_path).get_fdata())[0]
    if seg_path is None:
        return raw
    seg = np.asarray(nib.load(seg_path).get_fdata())[0]
    return raw, seg


def slice_labels(seg):
    """Non-zero labels, as MaskSplit does."""
    return [lab for lab in np.unique(seg) if lab != 0]


def read_manifest(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            for k in ("row", "H", "W", "zeroshot", "n_labels"):
                r[k] = int(r[k])
            rows.append(r)
    return rows


def cond_name(encoder, pipeline, noise_std=0.0):
    name = f"{encoder}__{pipeline}"
    if noise_std:
        name += f"__noise{noise_std:g}"
    return name


def parse_cond(name):
    """'sam_vit_b__medsam__noise1' -> ('sam_vit_b', 'medsam', 1.0)."""
    parts = name.split("__")
    noise = float(parts[2][len("noise"):]) if len(parts) > 2 else 0.0
    return parts[0], parts[1], noise


# ---------------------------------------------------------------------------
# Intensity preprocessing
# ---------------------------------------------------------------------------
def minmax_uint8_3c(raw):
    """Exactly utils/dataloader.NiiDataset._preprocess (H, W) -> (H, W, 3) uint8."""
    np_3c = np.stack([raw, raw, raw], axis=-1).astype(np.float64)
    np_3c = (np_3c - np_3c.min()) / (np_3c.max() - np_3c.min() + 1e-8)
    return np.round(np_3c * 255).astype(np.uint8)


def medsam_mr_preprocess(img_np):
    """Exactly medsam_mr_preprocess() in evaluation/test_medsam.py."""
    image_data = img_np.astype(np.float64)
    foreground = image_data[image_data > 0]
    if foreground.size == 0:
        return np.zeros_like(image_data, dtype=np.uint8)
    lower_bound, upper_bound = np.percentile(foreground, 0.5), np.percentile(foreground, 99.5)
    image_data_pre = np.clip(image_data, lower_bound, upper_bound)
    image_data_pre = (
        (image_data_pre - np.min(image_data_pre))
        / max(np.max(image_data_pre) - np.min(image_data_pre), 1e-8)
        * 255.0
    )
    image_data_pre[image_data == 0] = 0
    return np.uint8(image_data_pre)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def make_geom(pipeline, H, W):
    """(H, W, h, w): native size and the size of the image content inside the
    1024x1024 input (h, w < 1024 for 'sam' when the slice is not square)."""
    if pipeline == "sam":  # = ResizeLongestSide.get_preprocess_shape(H, W, IMG_SIZE)
        scale = IMG_SIZE * 1.0 / max(H, W)
        return (H, W, int(H * scale + 0.5), int(W * scale + 0.5))
    if pipeline == "medsam":
        return (H, W, IMG_SIZE, IMG_SIZE)
    raise ValueError(pipeline)


def preprocess(raw, pipeline, noise_std=0.0, rng=None):
    """Raw slice -> encoder input (3, 1024, 1024) float32 tensor, geometry.

    noise_std (grey levels on the 0-255 scale) adds Gaussian noise before the
    final normalisation; only used for the CKA noise-ceiling reference."""
    H, W = raw.shape
    if pipeline == "sam":
        from segment_anything.utils.transforms import ResizeLongestSide
        img = minmax_uint8_3c(raw)
        img = ResizeLongestSide(IMG_SIZE).apply_image(img).astype(np.float32)  # (h, w, 3)
        if noise_std:
            img = img + rng.normal(0.0, noise_std, img.shape).astype(np.float32)
        img = (img - SAM_PIXEL_MEAN) / SAM_PIXEL_STD
        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        h, w = x.shape[-2:]
        x = F.pad(x, (0, IMG_SIZE - w, 0, IMG_SIZE - h))  # same as Sam.preprocess
        return x, (H, W, h, w)
    if pipeline == "medsam":
        from skimage import transform as sktransform
        img = medsam_mr_preprocess(raw)
        img_3c = np.repeat(img[:, :, None], 3, axis=-1)
        r = sktransform.resize(img_3c, (IMG_SIZE, IMG_SIZE), order=3,
                               preserve_range=True, anti_aliasing=True).astype(np.uint8)
        r = r.astype(np.float32)
        if noise_std:
            r = r + rng.normal(0.0, noise_std, r.shape).astype(np.float32)
        r = (r - r.min()) / np.clip(r.max() - r.min(), a_min=1e-8, a_max=None)
        x = torch.from_numpy(np.ascontiguousarray(r.transpose(2, 0, 1)))
        return x, (H, W, IMG_SIZE, IMG_SIZE)
    raise ValueError(pipeline)


def mask_to_grid(mask, geom, grid=GRID):
    """Binary native mask (H, W) -> fraction of each token covered, (grid, grid)."""
    H, W, h, w = geom
    m = torch.from_numpy(mask.astype(np.float32))[None, None]
    m = F.interpolate(m, size=(h, w), mode="nearest")
    m = F.pad(m, (0, IMG_SIZE - w, 0, IMG_SIZE - h))
    return F.avg_pool2d(m, IMG_SIZE // grid)[0, 0].numpy()


def valid_grid(geom, grid=GRID):
    """Tokens that contain image content (False = padding only)."""
    H, W, h, w = geom
    return mask_to_grid(np.ones((H, W), bool), geom, grid) > 0


def grid_to_native(prob, geom):
    """Score map on any square grid (e.g. 64x64 probe output, 256x256 SAM
    low-res logits) -> native (H, W), the way SAM's postprocess_masks (pad:
    upsample to 1024, crop, resize) and MedSAM's inference (stretch: resize)
    do it. Bilinear, align_corners=False."""
    H, W, h, w = geom
    t = torch.as_tensor(prob, dtype=torch.float32)[None, None]
    if (h, w) != (IMG_SIZE, IMG_SIZE):
        t = F.interpolate(t, (IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        t = t[..., :h, :w]
    t = F.interpolate(t, (H, W), mode="bilinear", align_corners=False)
    return t[0, 0].numpy()


def box_from_mask(mask):
    """Tight box [x0, y0, x1, y1] in native pixels (gen_bboxes with jitter=0)."""
    ys, xs = np.nonzero(mask)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float64)


def box_to_input(box, geom):
    """Native box -> 1024 input frame (as SamPredictor / test_medsam.py do)."""
    H, W, h, w = geom
    return np.asarray(box, np.float64) * np.array([w / W, h / H, w / W, h / H])


def dice(a, b):
    a, b = a.astype(bool), b.astype(bool)
    s = a.sum() + b.sum()
    return float(2.0 * np.logical_and(a, b).sum() / s) if s else float("nan")


# ---------------------------------------------------------------------------
# Token sampling for CKA (identical for every encoder within one pipeline)
# ---------------------------------------------------------------------------
def token_positions(boxes_grid, valid, n_tokens, rng):
    """Half the tokens from inside the union of the GT boxes, half uniformly
    from the valid (non-padding) area. Returns flat indices into the 64x64
    grid (int32) and an in-box flag per token."""
    inbox = np.zeros_like(valid)
    for gx0, gy0, gx1, gy1 in boxes_grid:
        inbox[gy0:gy1 + 1, gx0:gx1 + 1] = True
    inbox &= valid
    box_idx = np.flatnonzero(inbox)
    n_box = min(n_tokens // 2, box_idx.size)
    pick_box = rng.choice(box_idx, n_box, replace=False) if n_box else np.zeros(0, int)
    rest = np.setdiff1d(np.flatnonzero(valid), pick_box)
    pick_rest = rng.choice(rest, n_tokens - n_box, replace=False)
    pos = np.concatenate([pick_box, pick_rest]).astype(np.int32)
    return pos, inbox.ravel()[pos]


def box_to_grid(box_1024):
    """Box in the 1024 frame -> inclusive token index range [gx0, gy0, gx1, gy1]."""
    g = np.floor(np.asarray(box_1024) / PATCH).astype(int)
    return np.clip(g, 0, GRID - 1)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def load_model(encoder, ckpt_dir, device, seed=0):
    """Returns (sam_model or None, image_encoder). The full SAM model is kept
    for the decoder sanity check; the random encoder has no decoder."""
    from segment_anything import sam_model_registry
    from segment_anything.modeling import ImageEncoderViT
    reg, fname = ENCODERS[encoder]
    if fname is None:
        torch.manual_seed(seed)
        enc = ImageEncoderViT(  # same arguments as segment_anything.build_sam for vit_b
            depth=12, embed_dim=768, img_size=IMG_SIZE, mlp_ratio=4,
            norm_layer=lambda d: torch.nn.LayerNorm(d, eps=1e-6), num_heads=12,
            patch_size=PATCH, qkv_bias=True, use_rel_pos=True,
            global_attn_indexes=[2, 5, 8, 11], window_size=14, out_chans=256,
        )
        return None, enc.to(device).eval()
    path = os.path.join(ckpt_dir, fname)
    if not os.path.isfile(path):  # build_sam would otherwise prompt for a download
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    sam = sam_model_registry[reg](checkpoint=path).to(device).eval()
    return sam, sam.image_encoder


class LayerTap:
    """Captures the token map entering block 0 ('embed' = patch embedding +
    position embedding) and the output of every transformer block, each as
    (B, 64, 64, D). The neck output is the encoder's return value."""

    def __init__(self, encoder):
        self.out, self.handles = {}, []
        self.names = ["embed"] + [f"block{i:02d}" for i in range(len(encoder.blocks))]
        self.handles.append(encoder.blocks[0].register_forward_pre_hook(
            lambda m, inp: self.out.__setitem__("embed", inp[0])))
        for i, blk in enumerate(encoder.blocks):
            self.handles.append(blk.register_forward_hook(
                lambda m, inp, o, n=f"block{i:02d}": self.out.__setitem__(n, o)))

    def remove(self):
        for h in self.handles:
            h.remove()
