#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import nibabel as nib
from skimage import io as skio
from segment_anything import sam_model_registry, SamPredictor

# ============================================================
# Utilities: normalization + loaders
# ============================================================

def _to_uint8_255(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0,255] and cast to uint8 (robust to constant arrays)."""
    arr = np.asarray(arr)
    amin = float(arr.min())
    amax = float(arr.max())
    if amax > amin:
        out = np.rint((arr - amin) / (amax - amin) * 255.0).astype(np.uint8)
    else:
        out = np.zeros_like(arr, dtype=np.uint8)
    return out

def _ensure_hw3_from_gray(gray_hw: np.ndarray) -> np.ndarray:
    """Replicate a single-channel HxW image to HxWx3 uint8."""
    gray_hw = np.asarray(gray_hw)
    if gray_hw.ndim != 2:
        raise ValueError(f"Expected a 2D array (H,W), got shape {gray_hw.shape}.")
    return np.stack([gray_hw, gray_hw, gray_hw], axis=-1)

def _coerce_rgb_uint8(img: np.ndarray) -> np.ndarray:
    """
    Coerce HxW / HxWx1 / HxWx3 / HxWx4 into HxWx3 uint8 in [0,255],
    applying min-max normalization when needed.
    """
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[..., 0]
    if img.ndim == 2:
        return _ensure_hw3_from_gray(_to_uint8_255(img))
    if img.ndim == 3:
        if img.shape[-1] == 4:
            img = img[..., :3]  # drop alpha
        if img.shape[-1] != 3:
            raise ValueError(f"Unsupported channel count {img.shape[-1]} for image input.")
        if img.dtype != np.uint8 or img.min() < 0 or img.max() > 255:
            img = _to_uint8_255(img)
        else:
            img = img.astype(np.uint8, copy=False)
        return img
    raise ValueError(f"Unsupported image array with ndim={img.ndim}, shape={img.shape}.")

def load_file(path: str):
    """
    Load a single file (NIfTI .nii/.nii.gz OR standard image .png/.jpg/.tif),
    normalize like NiiDataset (min-max → [0,255]) and return (HxWx3) uint8.

    NIfTI:
      - Expected shape: (1,H,W) or (H,W). (H,W,1) is tolerated via squeeze.
      - Output: (H,W,3) uint8 (gray→RGB replication).
    Image:
      - Accepts HxW / HxWx1 / HxWx3 / HxWx4; coerces to HxWx3 uint8.
    """
    lower = path.lower()
    is_nii = lower.endswith(".nii") or lower.endswith(".nii.gz")

    if is_nii:
        img = nib.load(path)
        data = img.get_fdata()  # float64
        if data.ndim == 3:
            if data.shape[0] == 1:           # (1,H,W)
                sig = data[0, ...]
            else:
                if 1 in data.shape:
                    sig = np.squeeze(data)
                    if sig.ndim != 2:
                        raise ValueError(f"NIfTI squeeze did not yield 2D; got {sig.shape}.")
                else:
                    raise ValueError(f"NIfTI expected (1,H,W) or (H,W); got {data.shape}.")
        elif data.ndim == 2:
            sig = data
        else:
            raise ValueError(f"Unsupported NIfTI dimensionality {data.ndim}; need (1,H,W) or (H,W).")

        sig_u8 = _to_uint8_255(sig)
        rgb = _ensure_hw3_from_gray(sig_u8)
        info = f"NIfTI input {data.shape} → slice {sig.shape} → output {rgb.shape} (HxWxC)"
        return rgb, info

    # Standard image
    img = skio.imread(path)  # HxW / HxWx3 / HxWx4
    rgb = _coerce_rgb_uint8(img)
    info = f"Image input {np.asarray(img).shape} → output {rgb.shape} (HxWxC)"
    return rgb, info


# ============================================================
# Model load & inference
# ============================================================

def load_sam_model(checkpoint: str, model_type: str = "vit_b", device: str = "cuda"):
    """
    model_type ∈ {'vit_b','vit_h','samri'}; 'samri' maps to 'vit_b' backbone.
    """
    key = "vit_b" if model_type.lower() in ("vit_b", "samri") else "vit_h"
    model = sam_model_registry[key](checkpoint=checkpoint)
    model = model.to(device)
    model.eval()
    return model

def run_predict(
    predictor: SamPredictor,
    image_hwc_uint8: np.ndarray,
    box=None,
    point=None,
    point_label=1,
    multimask_output=False,
) -> np.ndarray:
    """
    Run SAM predictor and return a binary mask with shape (1,H,W) uint8 {0,1}.
    """
    H, W, _ = image_hwc_uint8.shape
    predictor.set_image(image_hwc_uint8)

    point_coords = None
    point_labels = None
    if point is not None:
        point_coords = np.array(point, dtype=np.float32)  # (1,2)
        point_labels = np.array([point_label], dtype=np.int32)     # (1,)

    box_arr = None
    if box is not None:
        box_arr = np.array(box, dtype=np.float32)[None, :]         # (1,4)

    # default to full-image box if neither prompt is provided
    if box_arr is None and point_coords is None:
        box_arr = np.array([[0, 0, W - 1, H - 1]], dtype=np.float32)

    pred_masks, _, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box_arr,
        multimask_output=multimask_output,
    )
    if pred_masks.ndim == 2:
        pred_1hw = pred_masks[None, ...]
    else:
        pred_1hw = pred_masks[:1, ...]  # take first
    pred_1hw = (pred_1hw > 0.5).astype(np.uint8)
    return pred_1hw  # (1,H,W)


# ============================================================
# Saving
# ============================================================

def save_mask_nii_gz(out_dir: str, base_name: str, mask_1hw: np.ndarray, affine=None, header=None) -> str:
    """
    Save mask as .nii.gz with shape [1,H,W]. Uses identity affine by default.
    """
    os.makedirs(out_dir, exist_ok=True)
    if affine is None:
        affine = np.eye(4)
    nii = nib.Nifti1Image(mask_1hw.astype(np.uint8), affine=affine, header=header)
    out_path = os.path.join(out_dir, f"{base_name}_pred.nii.gz")
    nib.save(nii, out_path)
    return out_path

def save_mask_png(out_dir: str, base_name: str, mask_1hw: np.ndarray) -> str:
    """
    Save grayscale PNG (H×W) of the predicted mask (0/1 → 0/255).
    """
    os.makedirs(out_dir, exist_ok=True)
    h, w = mask_1hw.shape[1], mask_1hw.shape[2]
    mask_hw = (mask_1hw.reshape(h, w).astype(np.uint8)) * 255
    out_path = os.path.join(out_dir, f"{base_name}_pred.png")
    skio.imsave(out_path, mask_hw, check_contrast=False)
    return out_path


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Infer a single NIfTI or image with SAM/SAMRI and save mask as .nii.gz (+optional PNG)."
    )
    p.add_argument("--input", "-i", required=True, help="Path to input (.nii/.nii.gz or .png/.jpg/.tif)")
    p.add_argument("--output", "-o", required=True, help="Output folder")
    p.add_argument("--checkpoint", "-c", required=True, help="Path to SAM/SAMRI checkpoint (.pth)")
    p.add_argument("--model-type", default="vit_b", choices=["vit_b", "vit_h", "samri"],
                   help="Backbone key for sam_model_registry (SAMRI usually uses vit_b).")
    p.add_argument("--device", default="cuda", help='Device, e.g., "cuda" or "cpu"')
    p.add_argument("--box", nargs=4, type=float, default=None, metavar=("X1", "Y1", "X2", "Y2"),
                   help="Optional bounding box prompt (pixel coords).")
    p.add_argument("--point", nargs=2, type=float, default=None, metavar=("X", "Y"),
                   help="Optional point prompt (pixel coords).")
    p.add_argument("--no-png", action="store_true", help="Do not write PNG (only .nii.gz).")
    return p.parse_args()

def clean_basename(inp_path: str) -> str:
    """
    Turn '/path/case001.nii.gz' -> 'case001'; '/path/img.png' -> 'img'
    """
    base = os.path.basename(inp_path)
    if base.endswith(".nii.gz"):
        base = base[:-7]  # strip '.nii.gz'
    else:
        base = os.path.splitext(base)[0]
    return base

def main():
    args = parse_args()

    # 1) Load & normalize input to HxWx3 uint8
    image_hwc_uint8, info = load_file(args.input)
    print(info)
    print(f"Prepared image for SAM: {image_hwc_uint8.shape} (HxWxC uint8)")

    # 2) Model
    model = load_sam_model(args.checkpoint, model_type=args.model_type, device=args.device)
    predictor = SamPredictor(model)

    # 3) Predict -> (1,H,W)
    pred_1hw = run_predict(
        predictor,
        image_hwc_uint8,
        box=args.box,
        point=args.point,
        multimask_output=False,
    )

    # 4) Save outputs
    base = clean_basename(args.input)
    nii_path = save_mask_nii_gz(args.output, base, pred_1hw, affine=np.eye(4), header=None)
    print(f"Saved NIfTI: {nii_path}")
    if not args.no_png:
        png_path = save_mask_png(args.output, base, pred_1hw)
        print(f"Saved PNG:   {png_path}")
    print("Done.")

if __name__ == "__main__":
    main()
