#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import nibabel as nib
from skimage import io as skio
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
from skimage import measure
import torch
import platform
if platform.system() == "Darwin":
    os.environ.setdefault("TK_SILENCE_DEPRECATION", "1")

# -----------------------------
# SAM / SAMRI
# -----------------------------
try:
    from segment_anything import sam_model_registry, SamPredictor
except Exception as e:
    raise RuntimeError("segment_anything is not installed. Please install it before running this GUI.") from e


# =============================
# Utils: devices & model loading
# =============================
def pick_device(preferred: str = None) -> str:
    preferred = (preferred or "").lower()
    if preferred == "cuda" and torch.cuda.is_available():
        return "cuda"
    if preferred == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    # auto
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_sam_model_safe(checkpoint: str, model_type: str = "vit_b", device: str = None):
    """
    Build SAM model, load weights on CPU to avoid CUDA-only checkpoints,
    then move to chosen device (cuda/mps/cpu).
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    dev = pick_device(device)
    key = "vit_b" if model_type.lower() in ("vit_b", "samri") else "vit_h"

    model = sam_model_registry[key](checkpoint=None)
    ckpt = torch.load(checkpoint, map_location=torch.device("cpu"))
    if isinstance(ckpt, dict) and any(k in ckpt for k in ("state_dict", "model")):
        ckpt = ckpt.get("state_dict", ckpt.get("model", ckpt))

    # strip DataParallel "module." prefix if present
    if isinstance(ckpt, dict):
        ckpt = { (k[7:] if k.startswith("module.") else k): v for k, v in ckpt.items() }

    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing:    print(f"[SAM] Missing keys: {len(missing)} (showing up to 5): {missing[:5]}")
    if unexpected: print(f"[SAM] Unexpected keys: {len(unexpected)} (up to 5): {unexpected[:5]}")

    model.to(dev)
    model.eval()
    return model, dev


# =============================
# Utils: IO and normalization
# =============================

# --- macOS-safe filetypes ---
OPEN_FILETYPES = [
    ("NIfTI", (".nii", ".nii.gz")),
    ("Images", (".png", ".jpg", ".jpeg", ".tif", ".tiff")),
    ("All files", ("*",)),
]

SAVE_FILETYPES = [
    ("NIfTI (.nii.gz)", (".nii.gz",)),
    ("PNG (.png)", (".png",)),
    ("All files", ("*",)),
]

def _to_uint8_255(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    a, b = float(arr.min()), float(arr.max())
    if b > a:
        return np.rint((arr - a) / (b - a) * 255.0).astype(np.uint8)
    return np.zeros_like(arr, dtype=np.uint8)

def _ensure_hw3_from_gray(gray_hw: np.ndarray) -> np.ndarray:
    if gray_hw.ndim != 2:
        raise ValueError(f"Expected (H,W), got {gray_hw.shape}")
    return np.stack([gray_hw, gray_hw, gray_hw], axis=-1)

def _coerce_rgb_uint8(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[-1] == 1: img = img[..., 0]
    if img.ndim == 2:  return _ensure_hw3_from_gray(_to_uint8_255(img))
    if img.ndim == 3:
        if img.shape[-1] == 4: img = img[..., :3]
        if img.shape[-1] != 3: raise ValueError(f"Unsupported channels: {img.shape[-1]}")
        if img.dtype != np.uint8 or img.min() < 0 or img.max() > 255:
            img = _to_uint8_255(img)
        else:
            img = img.astype(np.uint8, copy=False)
        return img
    raise ValueError(f"Unsupported image array with ndim={img.ndim}, shape={img.shape}.")

def load_nii_or_img_as_rgb_uint8(path: str):
    """
    Load NIfTI (.nii/.nii.gz) expecting (1,H,W) or (H,W), OR standard image (.png/.jpg/.tif).
    Return: rgb(H,W,3) uint8, info string, and (affine, header) for NIfTI else (None, None).
    """
    lower = path.lower()
    is_nii = lower.endswith(".nii") or lower.endswith(".nii.gz")
    if is_nii:
        img = nib.load(path)
        data = img.get_fdata()
        if data.ndim == 3:
            if data.shape[0] == 1:
                sig = data[0, ...]
            elif 1 in data.shape:
                sig = np.squeeze(data)
                if sig.ndim != 2:
                    raise ValueError(f"NIfTI squeeze did not yield 2D; got {sig.shape}")
            else:
                raise ValueError(f"NIfTI expected (1,H,W) or (H,W); got {data.shape}")
        elif data.ndim == 2:
            sig = data
        else:
            raise ValueError(f"Unsupported NIfTI dim {data.ndim}")
        sig_u8 = _to_uint8_255(sig)
        rgb = _ensure_hw3_from_gray(sig_u8)
        info = f"NIfTI {data.shape} → {sig.shape} → {rgb.shape}"
        return rgb, info, img.affine, img.header
    # standard image
    arr = skio.imread(path)
    rgb = _coerce_rgb_uint8(arr)
    info = f"Image {np.asarray(arr).shape} → {rgb.shape}"
    return rgb, info, None, None


# =============================
# GUI App
# =============================
class SAMRIGUI:
    def __init__(self, root):
        self.root = root
        root.title("SAMRI GUI – box & point prompts")

        # State
        self.image_path = None
        self.rgb = None
        self.affine = None
        self.header = None
        self.checkpoint_path = None
        self.model = None
        self.predictor = None
        self.device = pick_device(None)
        self.box = None               # (x1,y1,x2,y2)
        self.points = []              # list of (x,y)
        self.dragging = False
        self.rect_artist = None
        self.overlay_im = None
        self.pred_mask = None         # (1,H,W) uint8

        # Top controls
        top = tk.Frame(root)
        top.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)

        tk.Button(top, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=4)
        self.image_label = tk.Label(top, text="(no image)")
        self.image_label.pack(side=tk.LEFT, padx=4)

        tk.Button(top, text="Pick Checkpoint", command=self.pick_checkpoint).pack(side=tk.LEFT, padx=8)
        self.ckpt_label = tk.Label(top, text="(no checkpoint)")
        self.ckpt_label.pack(side=tk.LEFT, padx=4)

        tk.Button(top, text="Load Model", command=self.load_model).pack(side=tk.LEFT, padx=8)
        self.device_label = tk.Label(top, text=f"Device: {self.device}")
        self.device_label.pack(side=tk.LEFT, padx=4)

        # Middle controls
        mid = tk.Frame(root)
        mid.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)
        tk.Button(mid, text="Generate Mask", command=self.generate_mask, bg="#d1ffd1").pack(side=tk.LEFT, padx=4)
        tk.Button(mid, text="Save Mask", command=self.save_mask).pack(side=tk.LEFT, padx=4)
        tk.Button(mid, text="Clear Prompts", command=self.clear_prompts).pack(side=tk.LEFT, padx=4)
        tk.Label(mid, text="(Drag: Box  •  Shift+Click: Point)").pack(side=tk.LEFT, padx=12)

        # Figure
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.ax.set_axis_off()
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Connect events
        self.cid_press   = self.canvas.mpl_connect("button_press_event", self.on_press)
        self.cid_release = self.canvas.mpl_connect("button_release_event", self.on_release)
        self.cid_motion  = self.canvas.mpl_connect("motion_notify_event", self.on_motion)

        # Instructions
        bottom = tk.Frame(root)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=6)
        msg = ("Instructions: Load an image, pick/load a checkpoint.\n"
               "- Drag LMB to draw a BOX.\n"
               "- Shift+LeftClick to add a POINT.\n"
               "- Click 'Generate Mask' to run SAMRI and overlay result.\n"
               "- 'Save Mask' lets you write .nii.gz or .png.")
        tk.Label(bottom, text=msg, justify=tk.LEFT).pack(anchor="w")

    # ----------- File ops -----------
    def load_image(self):
        try:
            path = filedialog.askopenfilename(
                title="Select Image",
                filetypes=OPEN_FILETYPES,
                parent=self.root
            )
        except Exception as e:
            messagebox.showerror("File dialog error", str(e))
            return
        if not path: return
        try:
            rgb, info, aff, hdr = load_nii_or_img_as_rgb_uint8(path)
        except Exception as e:
            messagebox.showerror("Load error", str(e))
            return
        self.image_path = path
        self.rgb = rgb
        self.affine = aff
        self.header = hdr
        self.image_label.config(text=os.path.basename(path))
        self.ax.clear()
        self.ax.imshow(rgb[..., 0], cmap="gray")
        self.ax.set_axis_off()
        self.canvas.draw()
        self.box = None
        self.points = []
        self.pred_mask = None
        self._remove_rect()
        self._remove_overlay()
        print("[Load]", info)

    def pick_checkpoint(self):
        try:
            path = filedialog.askopenfilename(
                title="Select SAM/SAMRI checkpoint (.pth)",
                filetypes=SAVE_FILETYPES,
                parent=self.root
            )
        except Exception as e:
            messagebox.showerror("Checkpoint error", str(e))
            return
        if not path: return
        self.checkpoint_path = path
        self.ckpt_label.config(text=os.path.basename(path))

    def load_model(self):
        if not self.checkpoint_path:
            messagebox.showwarning("Checkpoint", "Please pick a checkpoint first.")
            return
        try:
            self.model, self.device = load_sam_model_safe(self.checkpoint_path, model_type="vit_b", device=None)
            self.predictor = SamPredictor(self.model)
            self.device_label.config(text=f"Device: {self.device}")
            messagebox.showinfo("Model", f"Model loaded on {self.device}.")
        except Exception as e:
            messagebox.showerror("Model load error", str(e))

    # ----------- Mouse events -----------
    def on_press(self, event):
        if self.rgb is None: return
        if event.inaxes != self.ax: return
        # Shift+click adds a point
        if event.key == "shift":
            self.points.append((event.xdata, event.ydata))
            self.ax.scatter([event.xdata], [event.ydata], s=60, c='cyan', marker='x', linewidths=2)
            self.canvas.draw_idle()
            return
        # Start box
        self.dragging = True
        self.box = [event.xdata, event.ydata, event.xdata, event.ydata]  # x1,y1,x2,y2
        self._draw_rect()

    def on_motion(self, event):
        if not self.dragging or self.rgb is None: return
        if event.inaxes != self.ax: return
        self.box[2], self.box[3] = event.xdata, event.ydata
        self._draw_rect()

    def on_release(self, event):
        if not self.dragging or self.rgb is None: return
        self.dragging = False
        if event.inaxes != self.ax: return
        self.box[2], self.box[3] = event.xdata, event.ydata
        # normalize box to x1< x2, y1< y2
        x1, y1, x2, y2 = self.box
        x1, x2 = sorted([x1, x2]); y1, y2 = sorted([y1, y2])
        self.box = [x1, y1, x2, y2]
        self._draw_rect()

    def _draw_rect(self):
        self._remove_rect()
        if self.box is None: return
        x1, y1, x2, y2 = self.box
        if any(v is None for v in (x1, y1, x2, y2)): return
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                         fill=False, linewidth=2, edgecolor='yellow')
        self.rect_artist = rect
        self.ax.add_patch(rect)
        self.canvas.draw_idle()

    def _remove_rect(self):
        if self.rect_artist is not None:
            self.rect_artist.remove()
            self.rect_artist = None
            self.canvas.draw_idle()

    def _remove_overlay(self):
        if self.overlay_im is not None:
            self.overlay_im.remove()
            self.overlay_im = None
            self.canvas.draw_idle()

    # ----------- Inference -----------
    def generate_mask(self):
        if self.rgb is None:
            messagebox.showwarning("Image", "Load an image first.")
            return
        if self.predictor is None:
            # lazy-load model by asking for checkpoint if not loaded
            if not self.checkpoint_path:
                messagebox.showwarning("Checkpoint", "Pick a checkpoint, then click 'Load Model'.")
                return
            self.load_model()
            if self.predictor is None:
                return

        H, W, _ = self.rgb.shape
        self.predictor.set_image(self.rgb)

        point_coords = None
        point_labels = None
        if len(self.points) > 0:
            arr = np.array(self.points, dtype=np.float32)
            point_coords = arr
            point_labels = np.ones((arr.shape[0],), dtype=np.int32)

        box_arr = None
        if self.box is not None:
            x1, y1, x2, y2 = self.box
            box_arr = np.array([[x1, y1, x2, y2]], dtype=np.float32)

        # default full image if no prompts
        if box_arr is None and point_coords is None:
            box_arr = np.array([[0, 0, W - 1, H - 1]], dtype=np.float32)

        try:
            pred_masks, _, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box_arr,
                multimask_output=False,
            )
        except Exception as e:
            messagebox.showerror("Predict error", str(e))
            return

        if pred_masks.ndim == 2:
            pred_1hw = pred_masks[None, ...]
        else:
            pred_1hw = pred_masks[:1, ...]
        pred_1hw = (pred_1hw > 0.5).astype(np.uint8)
        self.pred_mask = pred_1hw  # (1,H,W)

        # Overlay result in one color (green) with alpha
        self._remove_overlay()
        overlay = np.ma.masked_where(self.pred_mask.reshape(H, W) == 0,
                                     self.pred_mask.reshape(H, W))
        self.overlay_im = self.ax.imshow(overlay, cmap="Greens", alpha=0.5, interpolation="none")
        self.canvas.draw_idle()

    # ----------- Save -----------
    def save_mask(self):
        if self.pred_mask is None:
            messagebox.showwarning("Mask", "No mask to save. Click 'Generate Mask' first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save mask",
            defaultextension=".nii.gz",
            filetypes=[("NIfTI", "*.nii.gz"), ("PNG", "*.png"), ("All files", "*.*")]
        )
        if not path: return
        try:
            if path.lower().endswith(".nii.gz"):
                affine = self.affine if self.affine is not None else np.eye(4)
                nii = nib.Nifti1Image(self.pred_mask.astype(np.uint8), affine=affine, header=self.header)
                nib.save(nii, path)
            else:
                # save grayscale PNG (H×W) 0/255
                H, W = self.pred_mask.shape[1], self.pred_mask.shape[2]
                skio.imsave(path, (self.pred_mask.reshape(H, W) * 255).astype(np.uint8), check_contrast=False)
            messagebox.showinfo("Saved", f"Mask saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

    # ----------- misc -----------
    def clear_prompts(self):
        self.points = []
        self.box = None
        self._remove_rect()
        self.canvas.draw_idle()


def main():
    root = tk.Tk()
    app = SAMRIGUI(root)
    root.geometry("900x900")
    root.mainloop()

if __name__ == "__main__":
    main()
