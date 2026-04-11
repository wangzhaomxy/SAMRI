#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test comparison SAM-based models (MCP-MedSAM, SAMed, MedSA) on NIfTI datasets.

Dataset folder structure expected under each root path:
  <root>/
    subset1/testing/*.nii.gz
    subset2/testing/*.nii.gz
    ...

Pass one or two roots via --dataset-path:
  1 path  → results stored under "trained" key only.
  2 paths → first path → "trained", second path → "zero_shot".

Prompt strategy per model:
  mcp_medsam  - box prompt + CLIP features (TinyViT, 256 input)
  samed       - prompt-free LoRA forward (SAM ViT-B with LoRA, 512 input)
  medsa       - point prompt + adapter (SAM ViT-B with adapters, 1024 input)

All prompts are derived from the ground-truth mask, matching the protocol in test_vis.py.

Output pickle structure:
  {
    "trained":   { subset_name: [ { img_name, mask_name, labels,
                                     dice, hd, msd, pixel_count,
                                     area_percentage }, ... ], ... },
    "zero_shot": { subset_name: [ ... ], ... }   # only present with 2 paths
  }

Usage examples:
  # SAMed — trained + zero-shot in one run
python evaluation/test_comparison_models.py \
--model samed \
--ckpt-path /scratch/user/s4670484/comparison_ckpt_samri/SAMed/sam_vit_b_01ec64.pth \
--lora-ckpt /scratch/user/s4670484/comparison_ckpt_samri/SAMed/epoch_159.pth \
--dataset-path /scratch/user/s4670484/Datasets/SAMRI_train_test /scratch/user/s4670484/Datasets/Zeroshot/ \
--save-path /scratch/user/s4670484/Eval_results/SAMRI_comparison/samed.pkl \
--debug

  # MCP-MedSAM — single dataset root
python evaluation/test_comparison_models.py \
--model mcp_medsam \
--ckpt-path /scratch/user/s4670484/comparison_ckpt_samri/mcp_medsam/mcp_best.pth \
--dataset-path /scratch/user/s4670484/Datasets/SAMRI_train_test \
--save-path /scratch/user/s4670484/Eval_results/SAMRI_comparison/mcp_medsam.pkl \
--debug

  # Medical-SAM-Adapter
python evaluation/test_comparison_models.py \
--model medsa \
--ckpt-path /scratch/user/s4670484/comparison_ckpt_samri/medSA/sam_vit_b_01ec64.pth \
--adapter-ckpt /scratch/user/s4670484/comparison_ckpt_samri/medSA/Kidney_Tumor_sam_128.pth \
--dataset-path /scratch/user/s4670484/Datasets/SAMRI_train_test /scratch/user/s4670484/Datasets/Zeroshot/ \
--save-path /scratch/user/s4670484/Eval_results/SAMRI_comparison/medsa.pkl \
--debug

  # Debug: 2 samples per subset
  python evaluation/test_comparison_models.py \
    --model samed \
    --ckpt-path /scratch/user/s4670484/comparison_ckpt_samri/SAMed/sam_vit_b_01ec64.pth \
    --lora-ckpt /scratch/user/s4670484/comparison_ckpt_samri/SAMed/epoch_159.pth \
    --dataset-path /scratch/user/s4670484/Datasets/SAMRI_train_test /scratch/user/s4670484/Datasets/Zeroshot/ \
    --save-path /scratch/user/s4670484/Eval_results/SAMRI_comparison/samed_debug.pkl 
    --debug
"""

import os
import sys
import pickle
import types
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from pathlib import Path
from glob import glob
from tqdm import tqdm
from scipy.ndimage import zoom
from torch.utils.data import DataLoader

# ── SAMRI project root ────────────────────────────────────────────────────────
SAMRI_ROOT = Path(__file__).resolve().parent.parent   # .../SAMRI/
EVAL_DIR   = Path(__file__).resolve().parent           # .../SAMRI/evaluation/
sys.path.insert(0, str(SAMRI_ROOT))

from utils.dataloader import NiiDataset
from utils.utils import MaskSplit, gen_points, gen_bboxes
try:
    # Python 3.10+ environments (MCP-MedSAM, MedSA)
    from utils.losses import dice_similarity, sd_hausdorff_distance, sd_mean_surface_distance
except TypeError:
    # Python 3.8 fallback (SAMed conda env) — utils.losses uses X|Y union
    # syntax that requires Python 3.10+; define the three metrics inline.
    from scipy import ndimage as _ndimage

    def dice_similarity(y_true, y_pred, square=False, smooth=1e-10):
        intersection = np.sum(y_true * y_pred)
        sum_of_pred  = np.sum(np.square(y_pred) if square else y_pred)
        sum_of_true  = np.sum(np.square(y_true) if square else y_true)
        return (2.0 * intersection + smooth) / (sum_of_pred + sum_of_true + smooth)

    def _surface_distance(input1, input2, sampling=1, connectivity=1):
        i1   = np.atleast_1d(input1.astype(int))
        i2   = np.atleast_1d(input2.astype(int))
        conn = _ndimage.morphology.generate_binary_structure(i1.ndim, connectivity)
        S      = i1 - _ndimage.morphology.binary_erosion(i1, conn)
        Sprime = i2 - _ndimage.morphology.binary_erosion(i2, conn)
        dta = _ndimage.morphology.distance_transform_edt((1 - S),      sampling)
        dtb = _ndimage.morphology.distance_transform_edt((1 - Sprime), sampling)
        return np.concatenate([np.ravel(dta[Sprime != 0]), np.ravel(dtb[S != 0])])

    def sd_hausdorff_distance(input1, input2, sampling=1, connectivity=1):
        return _surface_distance(input1, input2, sampling, connectivity).max()

    def sd_mean_surface_distance(input1, input2, sampling=1, connectivity=1):
        return _surface_distance(input1, input2, sampling, connectivity).mean()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Test comparison SAM-based models against the SAMRI NIfTI dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", dest="model", required=True,
                   choices=["mcp_medsam", "samed", "medsa"],
                   help="Which comparison model to test.")
    p.add_argument("--dataset-path", dest="dataset_path", nargs="+", required=True,
                   metavar="PATH",
                   help="One or two dataset root paths. "
                        "Each root must contain subset folders with a testing/ sub-dir. "
                        "First path → 'trained', second path (optional) → 'zero_shot'.")
    p.add_argument("--ckpt-path", dest="ckpt_path", required=True,
                   help="SAM ViT-B base checkpoint (.pth) for samed/medsa, "
                        "or the full MCP-MedSAM checkpoint for mcp_medsam.")
    p.add_argument("--lora-ckpt", dest="lora_ckpt", default=None,
                   help="[samed] Path to SAMed LoRA parameters checkpoint (.pth).")
    p.add_argument("--adapter-ckpt", dest="adapter_ckpt", default=None,
                   help="[medsa] Path to Medical-SAM-Adapter fine-tuned checkpoint (.pth).")
    p.add_argument("--save-path", dest="save_path", required=True,
                   help="Output .pkl file path (e.g. results/samed.pkl).")
    p.add_argument("--device", dest="device", default="cuda",
                   choices=["cuda", "cpu", "mps"])
    p.add_argument("--lora-rank", dest="lora_rank", type=int, default=4,
                   help="[samed] LoRA rank; must match the trained checkpoint (default 4).")
    p.add_argument("--num-workers", dest="num_workers", type=int, default=4,
                   help="DataLoader num_workers (default 4; set 0 for MPS / debugging).")
    p.add_argument("--debug", action="store_true",
                   help="Debug mode: process only 2 samples per subset to verify the "
                        "pipeline runs end-to-end without errors.")
    args = p.parse_args()
    if len(args.dataset_path) > 2:
        p.error("--dataset-path accepts at most 2 paths (trained + zero-shot).")
    return args


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers (same convention as test_vis.py)
# ─────────────────────────────────────────────────────────────────────────────
def get_testing_paths(root: str) -> list:
    """Return sorted list of <subset>/testing/ paths under root.

    Only includes subsets that have an actual testing/ subdirectory so that
    empty results are never silently created.
    """
    testing_dirs = []
    for ds in sorted(glob(root.rstrip("/") + "/*")):
        if os.path.isdir(ds) and os.path.isdir(ds + "/testing"):
            testing_dirs.append(ds + "/testing/")
    return testing_dirs


# ─────────────────────────────────────────────────────────────────────────────
# Shared preprocessing helpers
# ─────────────────────────────────────────────────────────────────────────────
def _resize_longest_side(img: np.ndarray, size: int) -> np.ndarray:
    """Resize HxWx3 uint8 image so its longest side equals *size* (aspect preserved)."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    return cv2.resize(img, (int(w * scale + 0.5), int(h * scale + 0.5)),
                      interpolation=cv2.INTER_AREA)


def _pad_to_square(arr: np.ndarray, size: int) -> np.ndarray:
    """Zero-pad array to size×size (bottom/right padding only)."""
    h, w = arr.shape[:2]
    pad_h, pad_w = size - h, size - w
    if arr.ndim == 3:
        return np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)))
    return np.pad(arr, ((0, pad_h), (0, pad_w)))


# ─────────────────────────────────────────────────────────────────────────────
# 1. MCP-MedSAM  –  box prompt + CLIP features, 256-input
# ─────────────────────────────────────────────────────────────────────────────
class _MedSAMLite(torch.nn.Module):
    """Minimal MedSAM_Lite wrapper (matches MCP-MedSAM infer.py)."""
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super().__init__()
        self.image_encoder  = image_encoder
        self.mask_decoder   = mask_decoder
        self.prompt_encoder = prompt_encoder

    def postprocess_masks(self, masks, new_size, original_size):
        masks = masks[..., :new_size[0], :new_size[1]]
        return F.interpolate(masks, size=original_size, mode="bilinear", align_corners=False)


def load_mcp_medsam(ckpt: str, device: str):
    """
    TinyViT image encoder + MaskDecoder_F4 + PubMed-CLIP.
    Requires: pip install transformers
    Note: CLIPModel is loaded on CPU; move to 'device' if GPU RAM permits.
    """
    sys.path.insert(0, str(EVAL_DIR / "MCP-MedSAM-main"))
    from models import PromptEncoder, TwoWayTransformer, TinyViT, MaskDecoder_F4
    from transformers import CLIPModel, CLIPTokenizer

    image_encoder = TinyViT(
        img_size=256, in_chans=3,
        embed_dims=[64, 128, 160, 320],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        mlp_ratio=4., drop_rate=0., drop_path_rate=0.,
        use_checkpoint=False, mbconv_expand_ratio=4.,
        local_conv_size=3, layer_lr_decay=0.8,
    )
    prompt_encoder = PromptEncoder(
        embed_dim=256, image_embedding_size=(64, 64),
        input_image_size=(256, 256), mask_in_chans=16,
    )
    mask_decoder = MaskDecoder_F4(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(depth=2, embedding_dim=256,
                                      mlp_dim=2048, num_heads=8),
        modality=True, contents=True, transformer_dim=256,
        iou_head_depth=3, iou_head_hidden_dim=256,
    )
    model    = _MedSAMLite(image_encoder, mask_decoder, prompt_encoder)
    ckpt_d   = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(ckpt_d["model"])
    model    = model.to(device).eval()

    # CLIP runs on CPU to save GPU memory
    clip_model = CLIPModel.from_pretrained(
        "flaviagiammarino/pubmed-clip-vit-base-patch32", resume_download=True
    )
    clip_model.requires_grad_(False)
    tokenizer = CLIPTokenizer.from_pretrained(
        "openai/clip-vit-base-patch16", resume_download=True
    )
    return model, clip_model, tokenizer


@torch.no_grad()
def infer_mcp_medsam(
    model, clip_model, tokenizer,
    image_hwc: np.ndarray, bbox: np.ndarray, device: str,
) -> np.ndarray:
    """
    Preprocessing:
      - Resize longest side to 256, pad to 256×256, normalize to [0, 1].
      - Box prompt scaled to 256 coordinate space.
      - CLIP text feature: "MR Image" (category_idx = 1).
      - CLIP image feature from cropped bbox region.
    Returns binary mask (H, W) uint8 {0, 1}.
    """
    import torchvision.transforms as T

    H, W = image_hwc.shape[:2]

    # ── CLIP text features (MR modality) ─────────────────────────────────────
    tokens     = tokenizer("MR Image", max_length=tokenizer.model_max_length,
                           padding="max_length", truncation=True, return_tensors="pt").input_ids
    text_feat  = clip_model.get_text_features(tokens).detach()  # (1, 512)
    text_feat  = text_feat.unsqueeze(0).to(device)              # (1, 1, 512)
    cat_idx    = torch.tensor([1], device=device)               # MR = 1

    # ── CLIP image features from the bbox crop ────────────────────────────────
    x1, y1, x2, y2 = np.clip(bbox.astype(int), [0, 0, 0, 0], [W, H, W, H])
    if x2 <= x1 or y2 <= y1:                    # degenerate box → use whole image
        x1, y1, x2, y2 = 0, 0, W, H
    crop      = image_hwc[y1:y2, x1:x2, :]     # (ch, cw, 3) uint8

    to_t      = T.ToTensor()                    # scales to [0, 1]
    crops_64  = T.Resize([64,  64],  antialias=True)(to_t(crop)).unsqueeze(0).to(device)
    crop_224  = T.Resize([224, 224], antialias=True)(to_t(crop)).unsqueeze(0)
    clip_feat = clip_model.get_image_features(crop_224)        # (1, 512)
    clip_feat = clip_feat.unsqueeze(0).to(device)              # (1, 1, 512)

    # ── Image preprocessing: resize-longest-256, pad, normalize ──────────────
    img_256  = _resize_longest_side(image_hwc, 256).astype(np.float32)
    newh, neww = img_256.shape[:2]
    img_256  = (img_256 - img_256.min()) / np.clip(
        img_256.max() - img_256.min(), 1e-8, None
    )
    img_256p = _pad_to_square(img_256, 256)
    tensor   = torch.from_numpy(img_256p).permute(2, 0, 1).unsqueeze(0).float().to(device)

    emb = model.image_encoder(tensor)

    # ── Box scaled to 256 coordinate space ───────────────────────────────────
    ratio = 256 / max(H, W)
    box_256 = (bbox.astype(np.float32) * ratio)
    box_t   = torch.as_tensor(box_256[None, None, :], dtype=torch.float, device=device)

    sparse_emb, dense_emb = model.prompt_encoder(
        points=None, boxes=box_t, masks=None,
        features=clip_feat, crops=crops_64,
        text_features=text_feat, category_idx=cat_idx,
    )
    logits, _, _, _, _ = model.mask_decoder(
        image_embeddings=emb,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_emb,
        dense_prompt_embeddings=dense_emb,
        multimask_output=False,
    )
    pred = torch.sigmoid(model.postprocess_masks(logits, (newh, neww), (H, W)))
    return (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# 2. SAMed  –  prompt-free LoRA forward, 512-input
# ─────────────────────────────────────────────────────────────────────────────
def load_samed(ckpt: str, lora_ckpt: str, rank: int, device: str):
    """
    LoRA-adapted SAM ViT-B using SAMed-main/segment_anything (modified build).
    Temporarily injects SAMed-main into sys.path, then restores the original
    segment_anything modules so other models are unaffected.
    """
    samed_dir = str(EVAL_DIR / "SAMed-main")

    # Stash any already-loaded segment_anything modules
    _cached_sa = {k: v for k, v in sys.modules.items()
                  if k == "segment_anything" or k.startswith("segment_anything.")}
    for k in list(_cached_sa):
        del sys.modules[k]

    sys.path.insert(0, samed_dir)
    try:
        from segment_anything import sam_model_registry as samed_registry
        from sam_lora_image_encoder import LoRA_Sam

        # Auto-detect num_classes from the LoRA checkpoint so the mask decoder
        # head matches the saved weights.
        # mask_tokens.weight shape = [num_classes + 1, 256] in SAMed's decoder.
        lora_state = torch.load(lora_ckpt, map_location="cpu")
        lora_sam_dict = lora_state.get("model", lora_state)
        token_shape = lora_sam_dict.get(
            "sam.mask_decoder.mask_tokens.weight",
            lora_sam_dict.get("mask_decoder.mask_tokens.weight", None),
        )
        num_classes = (token_shape.shape[0] - 1) if token_shape is not None else 1
        print(f"  [samed] auto-detected num_classes={num_classes} from LoRA checkpoint")

        sam, _ = samed_registry["vit_b"](
            image_size=512, num_classes=num_classes,
            checkpoint=ckpt,
            pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1],
        )
        net = LoRA_Sam(sam, rank)
        net.load_lora_parameters(lora_ckpt)
        net = net.to(device).eval()
    finally:
        # Restore original segment_anything so other imports are clean
        sys.path.remove(samed_dir)
        for k in list(sys.modules.keys()):
            if k == "segment_anything" or k.startswith("segment_anything."):
                del sys.modules[k]
        sys.modules.update(_cached_sa)

    return net


@torch.no_grad()
def infer_samed(net, image_hwc: np.ndarray, device: str, _label: int) -> np.ndarray:
    """
    Preprocessing:
      - Extract grayscale (channel 0), normalize to [0, 1], zoom to 512×512.
      - Replicate to 3 channels; forward via SAMed's prompt-free LoRA path.

    SAMed is a multi-class model (num_classes channels). We take argmax over
    the class dimension to get a label map, then return (argmax == label) as
    the binary prediction for the requested GT label.

    Returns binary mask (H, W) uint8 {0, 1}.
    """
    from einops import repeat

    H, W  = image_hwc.shape[:2]
    # SAMed uses pixel_mean=[0,0,0], pixel_std=[1,1,1] — no normalisation.
    # The model expects raw [0, 255] float values, same as Synapse training.
    gray  = image_hwc[:, :, 0].astype(np.float32)        # [0, 255]
    if H != 512 or W != 512:
        gray = zoom(gray, (512 / H, 512 / W), order=3)
    gray  = np.clip(gray, 0.0, 255.0)                    # guard against bicubic overshoot

    tensor  = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).float().to(device)
    tensor  = repeat(tensor, "b c h w -> b (r c) h w", r=3)

    outputs  = net(tensor, multimask_output=False, image_size=512)
    logits   = outputs["masks"]                                       # (1, C, 512, 512)
    logits   = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
    # SAMed is trained with a fixed class vocabulary (e.g. Synapse organs) that
    # does not map to SAMRI GT label values.  We therefore evaluate the binary
    # foreground prediction (any non-background class predicted) against each
    # GT label, consistent with how MCP-MedSAM and MedSA are evaluated.
    argmax   = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()   # (H, W) int
    return (argmax != 0).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Medical-SAM-Adapter (MedSA)  –  point prompt + adapter, 1024-input
# ─────────────────────────────────────────────────────────────────────────────
def load_medsa(ckpt: str, adapter_ckpt, device: str):  # adapter_ckpt: str or None
    """
    SAM ViT-B with AdapterBlock modifications (Medical-SAM-Adapter-main/models/sam).
    Loads the base SAM weights from *ckpt*, then overlays adapter weights from
    *adapter_ckpt* (strict=False, so only adapter parameters need to be present).
    """
    sys.path.insert(0, str(EVAL_DIR / "Medical-SAM-Adapter-main"))
    from models.sam import sam_model_registry as medsa_registry

    # Minimal args namespace required by Medical-SAM-Adapter's build_sam
    msa_args = types.SimpleNamespace(
        image_size=1024,
        mod="sam_adpt",
        multimask_output=1,
        mid_dim=None,
        thd=False,
        chunk=None,
    )
    net = medsa_registry["vit_b"](msa_args, checkpoint=ckpt).to(device)

    if adapter_ckpt is not None:
        state = torch.load(adapter_ckpt, map_location="cpu")
        state_dict = state.get("state_dict", state)
        net.load_state_dict(state_dict, strict=False)

    return net.eval()


@torch.no_grad()
def infer_medsa(net, image_hwc: np.ndarray, mask_gt: np.ndarray, device: str) -> np.ndarray:
    """
    Preprocessing:
      - Resize to 1024×1024 (float, [0, 255]), then apply SAM preprocess().
        preprocess() normalises with pixel_mean/std and adds any needed padding
        (none here since the image is already exactly 1024×1024).
      - Point prompt derived from the GT mask center (scaled to 1024).
    Postprocessing:
      - net.postprocess_masks() upsamples the decoder's 256×256 low-res logits
        to 1024×1024 (the encoder input size), removes any padding (none in our
        case), then downsamples to the original (H, W). This is the standard SAM
        postprocessing path and is required because the decoder output is in the
        1024×1024 coordinate space, not in the 256×256 output space.
    Returns binary mask (H, W) uint8 {0, 1}.
    """
    import torchvision.transforms.functional as TF

    H, W = image_hwc.shape[:2]
    sz   = 1024

    # Resize to sz×sz (values remain in [0, 255] float before SAM normalisation)
    tensor = torch.from_numpy(image_hwc).permute(2, 0, 1).float()
    tensor = TF.resize(tensor, [sz, sz]).unsqueeze(0).to(device)
    # preprocess: normalise + pad. We resized directly to (sz, sz) with no
    # padding, so input_size == (sz, sz) for postprocess_masks below.
    tensor = net.preprocess(tensor)

    # Point from GT mask (center of mass), scaled to 1024
    pt_orig   = gen_points(mask_gt)   # [[x, y]] in original image coords
    pt_scaled = np.array(
        [[pt_orig[0][0] * sz / W, pt_orig[0][1] * sz / H]], dtype=np.float32
    )
    pt_coords = torch.tensor(pt_scaled, device=device).unsqueeze(0)  # (1, 1, 2)
    pt_labels = torch.ones(1, 1, dtype=torch.int, device=device)      # (1, 1)

    emb = net.image_encoder(tensor)
    sparse_emb, dense_emb = net.prompt_encoder(
        points=(pt_coords, pt_labels), boxes=None, masks=None
    )
    low_res_pred, _ = net.mask_decoder(
        image_embeddings=emb,
        image_pe=net.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_emb,
        dense_prompt_embeddings=dense_emb,
        multimask_output=False,
    )
    # low_res_pred: (1, 1, 256, 256) — decoder logits in 1024×1024 coord space.
    # postprocess_masks: upsample 256→1024, crop to input_size, resize to (H, W).
    pred = net.postprocess_masks(low_res_pred, input_size=(sz, sz), original_size=(H, W))
    pred = torch.sigmoid(pred)
    return (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop  (mirrors get_test_record_from_ds in utils/visual.py)
# ─────────────────────────────────────────────────────────────────────────────
def run_evaluation(infer_fn, test_loader, debug: bool = False) -> list:
    """
    Iterate over *test_loader* (NiiDataset with multi_mask=True, with_name=True),
    run *infer_fn(image_hwc, single_binary_mask)* for every mask label,
    compute DSC / HD / MSD, and return a list of per-image result dicts.

    When *debug* is True, only the first 2 samples are processed so you can
    quickly verify the pipeline runs end-to-end without errors.
    """
    records = []
    DEBUG_LIMIT = 2

    for i, (image, mask, img_path, mask_path) in enumerate(tqdm(test_loader)):
        if debug and i >= DEBUG_LIMIT:
            break

        image = image.squeeze(0).detach().cpu().numpy()   # (H, W, 3) uint8
        mask  = mask.squeeze(0).detach().cpu().numpy()    # (1, H, W) multi-label
        H, W  = mask.shape[-2:]

        # DataLoader wraps strings in tuples/lists when batch_size=1
        if isinstance(img_path,  (tuple, list)): img_path  = img_path[0]
        if isinstance(mask_path, (tuple, list)): mask_path = mask_path[0]

        if debug:
            print(f"  [debug] sample {i + 1}/{DEBUG_LIMIT}: {Path(img_path).name}")

        dice_l, hd_l, msd_l, pix_l, area_l, lbl_l = [], [], [], [], [], []

        for each_mask, label in MaskSplit(mask):
            pred = infer_fn(image, each_mask, int(label))

            dice_l.append(dice_similarity(each_mask, pred))
            hd_l.append(sd_hausdorff_distance(each_mask, pred))
            msd_l.append(sd_mean_surface_distance(each_mask, pred))
            pix_l.append(int(np.sum(each_mask)))
            area_l.append(float(np.sum(each_mask) / (H * W)))
            lbl_l.append(label)

        records.append({
            "img_name":        img_path,
            "mask_name":       mask_path,
            "labels":          lbl_l,
            "dice":            dice_l,
            "hd":              hd_l,
            "msd":             msd_l,
            "pixel_count":     pix_l,
            "area_percentage": area_l,
        })

        if debug:
            print(f"    labels={lbl_l}  dice={[f'{d:.3f}' for d in dice_l]}"
                  f"  hd={[f'{h:.2f}' for h in hd_l]}")

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    if args.debug:
        print("[DEBUG MODE] Only 2 samples per subset will be processed.")

    # ── Load model and build the inference callable ──────────────────────────
    print(f"[{args.model}] Loading model from: {args.ckpt_path}")

    if args.model == "mcp_medsam":
        model, clip_model, tokenizer = load_mcp_medsam(args.ckpt_path, args.device)
        def infer_fn(img, msk, _label):
            return infer_mcp_medsam(
                model, clip_model, tokenizer,
                img, gen_bboxes(msk, jitter=0), args.device,
            )

    elif args.model == "samed":
        if args.lora_ckpt is None:
            raise ValueError("--lora-ckpt is required for --model samed")
        net = load_samed(args.ckpt_path, args.lora_ckpt, args.lora_rank, args.device)
        def infer_fn(img, msk, label):
            return infer_samed(net, img, args.device, label)

    elif args.model == "medsa":
        net = load_medsa(args.ckpt_path, args.adapter_ckpt, args.device)
        def infer_fn(img, msk, _label):
            return infer_medsa(net, img, msk, args.device)

    # ── Map dataset paths to split labels ────────────────────────────────────
    split_labels = ["trained", "zero_shot"]
    splits = list(zip(split_labels, args.dataset_path))   # [(label, path), ...]

    # ── Helper: evaluate all subsets under one root ───────────────────────────
    def _eval_split(root: str, label: str) -> dict:
        test_dirs = get_testing_paths(root)
        if not test_dirs:
            raise FileNotFoundError(
                f"No subset/testing/ directories found under [{label}]: {root}"
            )
        print(f"\n[{label}] Found {len(test_dirs)} subset(s) in: {root}")
        for d in test_dirs:
            print(f"  {d}")

        split_record = {}
        for test_dir in test_dirs:
            subset_name = Path(test_dir).parts[-2]   # parts[-1]='testing', parts[-2]=subset_name
            print(f"\n  [{label}] Processing: {subset_name}")
            ds     = NiiDataset([test_dir], multi_mask=True, with_name=True)
            loader = DataLoader(ds, batch_size=1, num_workers=args.num_workers)
            split_record[subset_name] = run_evaluation(infer_fn, loader, debug=args.debug)
        return split_record

    # ── Run evaluation ────────────────────────────────────────────────────────
    final_record = {}
    for label, root in splits:
        final_record[label] = _eval_split(root, label)

    # ── Save results ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    with open(args.save_path, "wb") as f:
        pickle.dump(final_record, f)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\nResults saved → {args.save_path}")
    for label, subsets in final_record.items():
        n_imgs = sum(len(v) for v in subsets.values())
        print(f"  {label}: {len(subsets)} subset(s), {n_imgs} image(s) total")


if __name__ == "__main__":
    main()
