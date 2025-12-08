############################################################
# MRI Viewer + SAMRI inference integration (slice-level)
# - Multi-view MRI with Paint/Erase/Box/Point/Hand tools
# - SAMRI checkpoint loading (load_sam_model + SamPredictor)
# - Generate Mask button with multi-mask selection via thumbnails
############################################################

import os
import math
from io import BytesIO

import numpy as np
import SimpleITK as sitk
from PIL import Image

import torch
from inference import load_sam_model
from segment_anything import SamPredictor

from ipycanvas import MultiCanvas
from ipywidgets import (
    VBox, HBox, Button, ColorPicker, IntSlider,
    ToggleButtons, Checkbox, Label, Text, Dropdown,
    Image as WImage, HTML
)
from ipyfilechooser import FileChooser

############################################################
# Global state
############################################################

VIEW_SIZE = 256  # internal canvas resolution (fixed)

volume3d = None      # float32 (Z, Y, X)
mask3d = None        # uint8  (Z, Y, X)

dim_z = dim_y = dim_x = 0

axial_index = 0
coronal_index = 0
sagittal_index = 0

current_view = "axial"

drawing = False
last_canvas_x = None
last_canvas_y = None

box_drawing = False

# Prompts
# boxes: {view, slice, x0,y0,x1,y1 (canvas), ix0,iy0,ix1,iy1 (image coords)}
# points: {view, slice, x,y (canvas), ix,iy (image coords)}
box_prompts = []
point_prompts = []

overlay_color_hex = "#ff0000"

# MRI / mask I/O meta
image_sitk = None
current_image_path = None
last_save_dir = None

# SAMRI model state
current_model_path = None
sam_model = None
sam_predictor = None
sam_device = None

# Embedding cache: avoid re-running set_image if slice unchanged
last_sam_image_view = None
last_sam_image_slice_idx = None

# Multi-mask candidates
sam_candidates = None   # dict: {view, slice_idx, masks, scores}

# Zoom & pan state
zoom_slider_value_default = 100
axial_center_x = axial_center_y = 0.0
cor_center_x = cor_center_z = 0.0
sag_center_y = sag_center_z = 0.0
mip_angle = 0.0  # rotation angle for MIP in degrees


############################################################
# Canvas (3 layers: background, mask overlay, UI)
############################################################

main_canv = MultiCanvas(3, width=VIEW_SIZE, height=VIEW_SIZE)
main_bg = main_canv[0]
main_ol = main_canv[1]
main_ui = main_canv[2]

# Display size (CSS)
main_canv.layout.width = f"{VIEW_SIZE}px"
main_canv.layout.height = f"{VIEW_SIZE}px"


############################################################
# Widgets
############################################################

status_label = Label(
    value="Choose an MRI file → image loads automatically → view appears. Then choose a SAMRI checkpoint → model loads automatically."
)


# --- File chooser for MRI ---
fc = FileChooser(
    os.getcwd(),
    select_default=True,
    show_only_dirs=False
)
fc.title = "<b>Choose MRI file</b>"
fc.filter_pattern = [
    "*.nii", "*.nii.gz",
    "*.mha", "*.mhd",
    "*.nrrd",
    "*.dcm",
    "*.png", "*.jpg", "*.jpeg",
    "*.gz"
]

# --- File chooser for SAMRI checkpoint ---
fc_model = FileChooser(
    os.getcwd(),
    select_default=True,
    show_only_dirs=False
)
fc_model.title = "<b>Choose SAMRI checkpoint</b>"
fc_model.filter_pattern = [
    "*.pt", "*.pth", "*.ckpt", "*.onnx", "*.bin", "*.safetensors"
]

# --- File chooser for MASK (load previously saved mask) ---
fc_mask = FileChooser(
    os.getcwd(),
    select_default=True,
    show_only_dirs=False
)
fc_mask.title = "<b>Choose mask file</b>"
# Use the main 3D medical formats that save_mask() already supports
fc_mask.filter_pattern = [
    "*.nii", "*.nii.gz",
    "*.mhd", "*.mha",
    "*.nrrd",
    "*.dcm"
]

# Tool selector: Hand / Paint / Erase / Box / Point
tool_selector = ToggleButtons(
    options=['Hand', 'Paint', 'Erase', 'Box', 'Point'],
    value='Hand',
    description='Tool:'
)

brush_size_slider = IntSlider(
    value=1,
    min=1,
    max=40,
    step=1,
    description='Brush size:'
)

color_picker = ColorPicker(
    concise=False,
    description='Mask color:',
    value=overlay_color_hex
)

show_mask_checkbox = Checkbox(
    value=True,
    description='Show mask overlay'
)

axial_slider = IntSlider(
    value=0,
    min=0,
    max=0,
    step=1,
    description='Axial (Z):',
    disabled=True
)
coronal_slider = IntSlider(
    value=0,
    min=0,
    max=0,
    step=1,
    description='Coronal (Y):',
    disabled=True
)
sagittal_slider = IntSlider(
    value=0,
    min=0,
    max=0,
    step=1,
    description='Sagittal (X):',
    disabled=True
)

# --- Add +/- buttons for slice sliders ---

axial_minus_btn = Button(description="-", layout={'width': '40px'})
axial_plus_btn  = Button(description="+", layout={'width': '40px'})

coronal_minus_btn = Button(description="-", layout={'width': '40px'})
coronal_plus_btn  = Button(description="+", layout={'width': '40px'})

sagittal_minus_btn = Button(description="-", layout={'width': '40px'})
sagittal_plus_btn  = Button(description="+", layout={'width': '40px'})


zoom_slider = IntSlider(
    value=zoom_slider_value_default,
    min=0,
    max=400,
    step=10,
    description='Image Zoom (%):'
)

zoom_out_button = Button(description='Zoom -')
zoom_in_button  = Button(description='Zoom +')

clear_prompt_button = Button(
    description='Clear prompts',
    button_style=''
)

clear_mask_button = Button(
    description='Clear mask',
    button_style='warning'
)

save_mask_button = Button(
    description='Save mask',
    button_style='success'
)

# Generate Mask button (SAMRI inference)
generate_mask_button = Button(
    description='Generate Mask',
    button_style='info'
)

# Tiny panel to show candidate mask thumbnails
candidate_thumbs_box = HBox([])

# View buttons
axial_view_button = Button(description="Axial")
coronal_view_button = Button(description="Coronal")
sagittal_view_button = Button(description="Sagittal")
mip_view_button = Button(description="3D MIP")

# Canvas display size slider (CSS only)
canvas_size_slider = IntSlider(
    value=VIEW_SIZE,
    min=128,
    max=512,
    step=64,
    description='Canvas size:'
)

canvas_size_minus_button = Button(description='Size -')
canvas_size_plus_button  = Button(description='Size +')

# Save mask controls
save_dir_chooser = FileChooser(
    os.getcwd(),
    select_default=True,
    show_only_dirs=True
)
save_dir_chooser.title = "<b>Choose folder to save mask</b>"

save_filename_text = Text(
    value='',
    description='Name:',
    placeholder='mask3d'
)

save_format_dropdown = Dropdown(
    options=['.nii.gz', '.mhd', '.mha', '.nrrd', '.dcm', '.png'],
    value='.nii.gz',
    description='Format:'
)

############################################################
# Load status labels for MRI, model & mask
############################################################

def _status_html(kind: str, loaded: bool, name: str = "") -> str:
    color = "green" if loaded else "red"
    state = "loaded" if loaded else "unloaded"
    extra = f" ({name})" if (loaded and name) else ""
    return f"<span style='color:{color}; font-weight:bold;'>{kind}: {state}{extra}</span>"

mri_load_label   = HTML(value=_status_html("MRI",   loaded=False))
model_load_label = HTML(value=_status_html("SAMRI", loaded=False))
mask_load_label  = HTML(value=_status_html("Mask",  loaded=False))


############################################################
# Helper functions
############################################################

def log_status(msg: str):
    status_label.value = msg

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def get_zoom_factor():
    z = zoom_slider.value / 100.0
    return max(z, 0.01)

def prepare_volume(img_sitk_local):
    arr = sitk.GetArrayFromImage(img_sitk_local)

    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim == 3:
        pass
    elif arr.ndim == 4:
        arr = arr[0]
    else:
        raise ValueError(f"Unsupported dimensions: {arr.shape}")

    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, [2, 98])
    hi = max(hi, lo + 1e-6)
    arr = (arr - lo) / (hi - lo)
    arr = np.clip(arr, 0, 1)
    return arr

def init_mask():
    global mask3d
    if volume3d is None:
        mask3d = None
    else:
        mask3d = np.zeros_like(volume3d, dtype=np.uint8)

def split_name_and_ext(path):
    base = os.path.basename(path)
    if base.lower().endswith('.nii.gz'):
        return base[:-7], '.nii.gz'
    for ext in ['.nii', '.mhd', '.mha', '.nrrd', '.dcm', '.gz', '.png']:
        if base.lower().endswith(ext):
            return base[:-len(ext)], ext
    return base, ''


def get_view_window(view_name):
    """
    For zoom >= 1: crop window in slice coords.
    For zoom < 1: full slice, shrinking handled later.
    """
    global axial_center_x, axial_center_y
    global cor_center_x, cor_center_z
    global sag_center_y, sag_center_z

    zf = get_zoom_factor()

    if view_name == "axial":
        h, w = dim_y, dim_x
        cx, cy = axial_center_x, axial_center_y
    elif view_name == "coronal":
        h, w = dim_z, dim_x
        cx, cy = cor_center_x, cor_center_z
    elif view_name == "sagittal":
        h, w = dim_z, dim_y
        cx, cy = sag_center_y, sag_center_z
    else:
        return 0.0, 0.0, 0.0, 0.0

    if w == 0 or h == 0:
        return 0.0, 0.0, float(w), float(h)

    # zoom <=1: full slice
    if zf <= 1.0:
        if view_name == "axial":
            axial_center_x, axial_center_y = w / 2.0, h / 2.0
        elif view_name == "coronal":
            cor_center_x, cor_center_z = w / 2.0, h / 2.0
        elif view_name == "sagittal":
            sag_center_y, sag_center_z = w / 2.0, h / 2.0
        return 0.0, 0.0, float(w), float(h)

    width_view = w / zf
    height_view = h / zf
    width_view = min(width_view, w)
    height_view = min(height_view, h)

    left = cx - width_view / 2.0
    top = cy - height_view / 2.0

    if left < 0:
        left = 0.0
    if top < 0:
        top = 0.0
    if left + width_view > w:
        left = w - width_view
    if top + height_view > h:
        top = h - height_view

    new_cx = left + width_view / 2.0
    new_cy = top + height_view / 2.0

    if view_name == "axial":
        axial_center_x, axial_center_y = new_cx, new_cy
    elif view_name == "coronal":
        cor_center_x, cor_center_z = new_cx, new_cy
    elif view_name == "sagittal":
        sag_center_y, sag_center_z = new_cx, new_cy

    return left, top, width_view, height_view


############################################################
# Drawing slices (zoom < 1 shrinks & pads with black)
############################################################

def _draw_ui(view_name, slice_idx):
    main_ui.clear()
    if volume3d is None:
        return

    main_ui.stroke_style = 'yellow'
    main_ui.line_width = 2

    for box in box_prompts:
        if box['view'] == view_name and box['slice'] == slice_idx:
            x0,y0,x1,y1 = box['x0'], box['y0'], box['x1'], box['y1']
            x=min(x0,x1); y=min(y0,y1)
            w=abs(x1-x0); h=abs(y1-y0)
            main_ui.stroke_rect(x,y,w,h)

    main_ui.fill_style = 'cyan'
    for p in point_prompts:
        if p['view'] == view_name and p['slice'] == slice_idx:
            main_ui.begin_path()
            main_ui.arc(p['x'], p['y'], 4, 0, 2*math.pi)
            main_ui.fill()


def _draw_slice_zoomed(slice2d, mask_slice, view_name, slice_idx):
    """
    Common helper:
      - zoom <=1: shrink whole slice, center it, pad with black
      - zoom >1: crop via get_view_window, fill canvas
    """
    main_bg.clear()
    main_ol.clear()
    main_ui.clear()

    if volume3d is None:
        return

    zf = get_zoom_factor()

    if zf <= 1.0:
        # shrink full slice
        h, w = slice2d.shape
        img_size = int(VIEW_SIZE * zf)
        if img_size <= 0:
            return

        rgba = np.zeros((VIEW_SIZE, VIEW_SIZE, 4), dtype=np.uint8)
        rgba[..., 3] = 255  # opaque black

        img8 = (slice2d * 255).astype(np.uint8)
        img = Image.fromarray(img8).resize((img_size, img_size), Image.BILINEAR)
        rgb = np.array(img.convert("RGB"), dtype=np.uint8)

        offset = (VIEW_SIZE - img_size) // 2
        rgba[offset:offset+img_size, offset:offset+img_size, :3] = rgb

        main_bg.put_image_data(rgba, 0, 0)

        if show_mask_checkbox.value and mask3d is not None and mask_slice is not None:
            mask_img = Image.fromarray((mask_slice * 255).astype(np.uint8)).resize(
                (img_size, img_size), Image.NEAREST
            )
            mask_np = np.array(mask_img) > 0
            overlay = np.zeros((VIEW_SIZE, VIEW_SIZE, 4), dtype=np.uint8)
            r, g, b = hex_to_rgb(overlay_color_hex)
            overlay[offset:offset+img_size, offset:offset+img_size, 0][mask_np] = r
            overlay[offset:offset+img_size, offset:offset+img_size, 1][mask_np] = g
            overlay[offset:offset+img_size, offset:offset+img_size, 2][mask_np] = b
            overlay[offset:offset+img_size, offset:offset+img_size, 3][mask_np] = 120
            main_ol.put_image_data(overlay, 0, 0)

    else:
        # crop
        if view_name == "axial":
            h, w = dim_y, dim_x
        elif view_name == "coronal":
            h, w = dim_z, dim_x
        else:
            h, w = dim_z, dim_y

        left, top, width_view, height_view = get_view_window(view_name)

        x0 = int(max(0, min(w-1, math.floor(left))))
        x1 = int(max(x0+1, min(w, math.ceil(left + width_view))))
        y0 = int(max(0, min(h-1, math.floor(top))))
        y1 = int(max(y0+1, min(h, math.ceil(top + height_view))))

        sub = slice2d[y0:y1, x0:x1]
        img8 = (sub * 255).astype(np.uint8)
        img = Image.fromarray(img8).resize((VIEW_SIZE, VIEW_SIZE), Image.BILINEAR)
        rgb = np.array(img.convert("RGB"), dtype=np.uint8)

        rgba = np.zeros((VIEW_SIZE, VIEW_SIZE, 4), dtype=np.uint8)
        rgba[..., :3] = rgb
        rgba[..., 3] = 255
        main_bg.put_image_data(rgba, 0, 0)

        if show_mask_checkbox.value and mask3d is not None and mask_slice is not None:
            sub_mask = mask_slice[y0:y1, x0:x1]
            mask_img = Image.fromarray((sub_mask * 255).astype(np.uint8)).resize(
                (VIEW_SIZE, VIEW_SIZE), Image.NEAREST
            )
            mask_np = np.array(mask_img) > 0
            overlay = np.zeros((VIEW_SIZE, VIEW_SIZE, 4), dtype=np.uint8)
            r, g, b = hex_to_rgb(overlay_color_hex)
            overlay[mask_np, 0] = r
            overlay[mask_np, 1] = g
            overlay[mask_np, 2] = b
            overlay[mask_np, 3] = 120
            main_ol.put_image_data(overlay, 0, 0)

    _draw_ui(view_name, slice_idx)


def draw_axial():
    if volume3d is None:
        main_bg.clear(); main_ol.clear(); main_ui.clear()
        return
    slice2d = volume3d[axial_index]
    mask_slice = mask3d[axial_index] if mask3d is not None else None
    _draw_slice_zoomed(slice2d, mask_slice, "axial", axial_index)


def draw_coronal():
    if volume3d is None:
        main_bg.clear(); main_ol.clear(); main_ui.clear()
        return
    slice2d = volume3d[:, coronal_index, :]
    mask_slice = mask3d[:, coronal_index, :] if mask3d is not None else None
    _draw_slice_zoomed(slice2d, mask_slice, "coronal", coronal_index)


def draw_sagittal():
    if volume3d is None:
        main_bg.clear(); main_ol.clear(); main_ui.clear()
        return
    slice2d = volume3d[:, :, sagittal_index]
    mask_slice = mask3d[:, :, sagittal_index] if mask3d is not None else None
    _draw_slice_zoomed(slice2d, mask_slice, "sagittal", sagittal_index)


def draw_mip():
    main_bg.clear(); main_ol.clear(); main_ui.clear()
    if volume3d is None:
        return

    global mip_angle

    mip = volume3d.max(axis=0)
    img8 = (mip * 255).astype(np.uint8)
    img = Image.fromarray(img8)

    img = img.rotate(mip_angle, resample=Image.BILINEAR, expand=False)
    img = img.resize((VIEW_SIZE, VIEW_SIZE), Image.BILINEAR)
    rgb = np.array(img.convert("RGB"), dtype=np.uint8)

    rgba = np.zeros((VIEW_SIZE, VIEW_SIZE, 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = 255
    main_bg.put_image_data(rgba, 0, 0)

    if show_mask_checkbox.value and mask3d is not None:
        mip_mask = (mask3d > 0).max(axis=0)
        mask_img = Image.fromarray((mip_mask * 255).astype(np.uint8))
        mask_img = mask_img.rotate(mip_angle, resample=Image.NEAREST, expand=False)
        mask_img = mask_img.resize((VIEW_SIZE, VIEW_SIZE), Image.NEAREST)
        mask_np = np.array(mask_img) > 0

        overlay = np.zeros((VIEW_SIZE, VIEW_SIZE, 4), dtype=np.uint8)
        r, g, b = hex_to_rgb(overlay_color_hex)
        overlay[mask_np, 0] = r
        overlay[mask_np, 1] = g
        overlay[mask_np, 2] = b
        overlay[mask_np, 3] = 120
        main_ol.put_image_data(overlay, 0, 0)


def redraw():
    if current_view == "axial":
        draw_axial()
    elif current_view == "coronal":
        draw_coronal()
    elif current_view == "sagittal":
        draw_sagittal()
    else:
        draw_mip()


############################################################
# Canvas→image coordinate helpers (original resolution)
############################################################

def canvas_to_voxel_axial(cx, cy):
    if dim_x == 0 or dim_y == 0:
        return None, None

    zf = get_zoom_factor()

    if zf <= 1.0:
        img_size = int(VIEW_SIZE * zf)
        if img_size <= 0:
            return None, None
        offset = (VIEW_SIZE - img_size) / 2.0
        if not (offset <= cx < offset + img_size and offset <= cy < offset + img_size):
            return None, None
        u = (cx - offset) / img_size
        v = (cy - offset) / img_size
        vx = int(np.clip(round(u * (dim_x - 1)), 0, dim_x - 1))
        vy = int(np.clip(round(v * (dim_y - 1)), 0, dim_y - 1))
        return vx, vy
    else:
        left, top, width_view, height_view = get_view_window("axial")
        vx = left + (cx / VIEW_SIZE) * width_view
        vy = top + (cy / VIEW_SIZE) * height_view
        vx = int(np.clip(round(vx), 0, dim_x - 1))
        vy = int(np.clip(round(vy), 0, dim_y - 1))
        return vx, vy


def canvas_to_voxel_coronal(cx, cy):
    if dim_x == 0 or dim_z == 0:
        return None, None

    zf = get_zoom_factor()

    if zf <= 1.0:
        img_size = int(VIEW_SIZE * zf)
        if img_size <= 0:
            return None, None
        offset = (VIEW_SIZE - img_size) / 2.0
        if not (offset <= cx < offset + img_size and offset <= cy < offset + img_size):
            return None, None
        u = (cx - offset) / img_size
        v = (cy - offset) / img_size
        vx = int(np.clip(round(u * (dim_x - 1)), 0, dim_x - 1))
        vz = int(np.clip(round(v * (dim_z - 1)), 0, dim_z - 1))
        return vz, vx
    else:
        left, top, width_view, height_view = get_view_window("coronal")
        vx = left + (cx / VIEW_SIZE) * width_view
        vz = top + (cy / VIEW_SIZE) * height_view
        vx = int(np.clip(round(vx), 0, dim_x - 1))
        vz = int(np.clip(round(vz), 0, dim_z - 1))
        return vz, vx


def canvas_to_voxel_sagittal(cx, cy):
    if dim_y == 0 or dim_z == 0:
        return None, None

    zf = get_zoom_factor()

    if zf <= 1.0:
        img_size = int(VIEW_SIZE * zf)
        if img_size <= 0:
            return None, None
        offset = (VIEW_SIZE - img_size) / 2.0
        if not (offset <= cx < offset + img_size and offset <= cy < offset + img_size):
            return None, None
        u = (cx - offset) / img_size
        v = (cy - offset) / img_size
        vy = int(np.clip(round(u * (dim_y - 1)), 0, dim_y - 1))
        vz = int(np.clip(round(v * (dim_z - 1)), 0, dim_z - 1))
        return vz, vy
    else:
        left, top, width_view, height_view = get_view_window("sagittal")
        vy = left + (cx / VIEW_SIZE) * width_view
        vz = top + (cy / VIEW_SIZE) * height_view
        vy = int(np.clip(round(vy), 0, dim_y - 1))
        vz = int(np.clip(round(vz), 0, dim_z - 1))
        return vz, vy


def canvas_to_image_xy(view_name, cx, cy):
    """
    Map canvas coords to (ix, iy) in the *original* 2D slice:
      axial:    slice shape (Y, X)  → (ix=X, iy=Y)
      coronal:  slice shape (Z, X)  → (ix=X, iy=Z)
      sagittal: slice shape (Z, Y)  → (ix=Y, iy=Z)
    """
    if view_name == "axial":
        vx, vy = canvas_to_voxel_axial(cx, cy)
        if vx is None: return None, None
        return vx, vy
    elif view_name == "coronal":
        vz, vx = canvas_to_voxel_coronal(cx, cy)
        if vz is None: return None, None
        return vx, vz
    elif view_name == "sagittal":
        vz, vy = canvas_to_voxel_sagittal(cx, cy)
        if vz is None: return None, None
        return vy, vz
    else:
        return None, None


############################################################
# Painting (uses voxel coords)
############################################################

def paint_circle(cx, cy):
    if mask3d is None:
        return

    tool = tool_selector.value

    if tool not in ("Paint", "Erase"):
        return

    if current_view == "axial":
        vx, vy = canvas_to_voxel_axial(cx, cy)
        if vx is None:
            return
        z = axial_index
        scale = (dim_x/VIEW_SIZE + dim_y/VIEW_SIZE)/2
        r_vox = max(1, int(brush_size_slider.value * scale))

        y0=max(0,vy-r_vox); y1=min(dim_y, vy+r_vox+1)
        x0=max(0,vx-r_vox); x1=min(dim_x, vx+r_vox+1)

        yy,xx = np.ogrid[y0:y1, x0:x1]
        region = (yy-vy)**2 + (xx-vx)**2 <= r_vox*r_vox

        if tool == "Paint":
            mask3d[z,y0:y1,x0:x1][region]=1
        elif tool == "Erase":
            mask3d[z,y0:y1,x0:x1][region]=0

    elif current_view == "coronal":
        vz, vx = canvas_to_voxel_coronal(cx, cy)
        if vz is None:
            return
        y = coronal_index
        scale = (dim_x/VIEW_SIZE + dim_z/VIEW_SIZE)/2
        r_vox = max(1, int(brush_size_slider.value*scale))

        z0=max(0,vz-r_vox); z1=min(dim_z, vz+r_vox+1)
        x0=max(0,vx-r_vox); x1=min(dim_x, vx+r_vox+1)

        zz,xx = np.ogrid[z0:z1, x0:x1]
        region = (zz-vz)**2 + (xx-vx)**2 <= r_vox*r_vox

        if tool == "Paint":
            mask3d[z0:z1,y,x0:x1][region]=1
        elif tool == "Erase":
            mask3d[z0:z1,y,x0:x1][region]=0

    elif current_view == "sagittal":
        vz, vy = canvas_to_voxel_sagittal(cx, cy)
        if vz is None:
            return
        x = sagittal_index
        scale = (dim_y/VIEW_SIZE + dim_z/VIEW_SIZE)/2
        r_vox = max(1, int(brush_size_slider.value*scale))

        z0=max(0,vz-r_vox); z1=min(dim_z, vz+r_vox+1)
        y0=max(0,vy-r_vox); y1=min(dim_y, vy+r_vox+1)

        zz,yy=np.ogrid[z0:z1, y0:y1]
        region = (zz-vz)**2 + (yy-vy)**2 <= r_vox*r_vox

        if tool == "Paint":
            mask3d[z0:z1,y0:y1,x][region]=1
        elif tool == "Erase":
            mask3d[z0:z1,y0:y1,x][region]=0

    redraw()


def paint_line(x0,y0,x1,y1):
    steps = int(max(abs(x1-x0),abs(y1-y0))/max(brush_size_slider.value/2,1))+1
    for t in np.linspace(0,1,steps):
        paint_circle(x0+t*(x1-x0), y0+t*(y1-y0))


############################################################
# Hand tool (pan / rotate MIP)
############################################################

def pan_view(view_name, dx_canvas, dy_canvas):
    global axial_center_x, axial_center_y
    global cor_center_x, cor_center_z
    global sag_center_y, sag_center_z

    if zoom_slider.value <= 100:
        return

    left, top, width_view, height_view = get_view_window(view_name)
    if width_view <= 0 or height_view <= 0:
        return

    if view_name == "axial":
        dx_slice = -dx_canvas / VIEW_SIZE * width_view
        dy_slice = -dy_canvas / VIEW_SIZE * height_view
        axial_center_x += dx_slice
        axial_center_y += dy_slice

    elif view_name == "coronal":
        dx_slice = -dx_canvas / VIEW_SIZE * width_view
        dy_slice = -dy_canvas / VIEW_SIZE * height_view
        cor_center_x += dx_slice
        cor_center_z += dy_slice

    elif view_name == "sagittal":
        dx_slice = -dx_canvas / VIEW_SIZE * width_view
        dy_slice = -dy_canvas / VIEW_SIZE * height_view
        sag_center_y += dx_slice
        sag_center_z += dy_slice


############################################################
# Mouse events
############################################################

def get_current_slice():
    if current_view == "axial":
        return axial_index
    if current_view == "coronal":
        return coronal_index
    if current_view == "sagittal":
        return sagittal_index
    return axial_index

def on_mouse_down(x,y):
    global drawing, last_canvas_x, last_canvas_y, box_drawing

    if volume3d is None:
        return

    tool = tool_selector.value

    if current_view=="mip" and tool in ("Paint", "Erase", "Box", "Point"):
        return

    if tool in ("Paint", "Erase"):
        drawing=True
        last_canvas_x, last_canvas_y = x,y
        paint_circle(x,y)

    elif tool=="Box":
        slice_idx = get_current_slice()
        ix0, iy0 = canvas_to_image_xy(current_view, x, y)
        if ix0 is None:
            return
        box_prompts.append({
            'view': current_view,
            'slice': slice_idx,
            'x0':x, 'y0':y, 'x1':x, 'y1':y,        # canvas coords
            'ix0':ix0, 'iy0':iy0, 'ix1':ix0, 'iy1':iy0  # image coords
        })
        box_drawing=True
        _draw_ui(current_view, slice_idx)

    elif tool=="Point":
        slice_idx=get_current_slice()
        ix, iy = canvas_to_image_xy(current_view, x, y)
        if ix is None:
            return
        point_prompts.append({
            'view':current_view,
            'slice':slice_idx,
            'x':x,'y':y,      # canvas
            'ix':ix,'iy':iy   # image
        })
        _draw_ui(current_view, slice_idx)

    elif tool=="Hand":
        drawing = True
        last_canvas_x, last_canvas_y = x, y


def on_mouse_move(x,y):
    global drawing, box_drawing, last_canvas_x, last_canvas_y, mip_angle
    if volume3d is None:
        return

    tool = tool_selector.value

    if tool in ("Paint", "Erase") and drawing and current_view != "mip":
        paint_line(last_canvas_x, last_canvas_y, x, y)
        last_canvas_x, last_canvas_y = x,y

    elif tool=="Box" and box_drawing and current_view != "mip":
        if box_prompts:
            box_prompts[-1]['x1']=x
            box_prompts[-1]['y1']=y
            ix1, iy1 = canvas_to_image_xy(current_view, x, y)
            if ix1 is not None:
                box_prompts[-1]['ix1'] = ix1
                box_prompts[-1]['iy1'] = iy1
            _draw_ui(current_view, get_current_slice())

    elif tool=="Hand" and drawing:
        dx = x - last_canvas_x
        dy = y - last_canvas_y

        if current_view == "mip":
            mip_angle += dx * 0.3
            last_canvas_x, last_canvas_y = x, y
            redraw()
        else:
            pan_view(current_view, dx, dy)
            last_canvas_x, last_canvas_y = x, y
            redraw()


def on_mouse_up(x,y):
    global drawing, box_drawing
    if volume3d is None:
        drawing=False; box_drawing=False
        return

    tool=tool_selector.value

    if tool in ("Paint", "Erase"):
        drawing=False

    elif tool=="Box" and box_drawing:
        if current_view != "mip" and box_prompts:
            box_prompts[-1]['x1']=x
            box_prompts[-1]['y1']=y
            ix1, iy1 = canvas_to_image_xy(current_view, x, y)
            if ix1 is not None:
                box_prompts[-1]['ix1'] = ix1
                box_prompts[-1]['iy1'] = iy1
            _draw_ui(current_view, get_current_slice())
        box_drawing=False

    elif tool=="Hand":
        drawing=False


main_canv.on_mouse_down(on_mouse_down)
main_canv.on_mouse_move(on_mouse_move)
main_canv.on_mouse_up(on_mouse_up)


############################################################
# Load MRI (now auto-triggered from file chooser)
############################################################

def load_from_path(_chooser):
    global volume3d, mask3d
    global dim_z, dim_y, dim_x
    global axial_index, coronal_index, sagittal_index
    global box_prompts, point_prompts, drawing, box_drawing, current_view
    global image_sitk, current_image_path, last_save_dir
    global axial_center_x, axial_center_y
    global cor_center_x, cor_center_z
    global sag_center_y, sag_center_z
    global mip_angle
    global last_sam_image_view, last_sam_image_slice_idx
    global sam_candidates

    path = fc.selected
    if not path:
        log_status("Please choose an MRI file first.")
        return

    path = os.path.abspath(path)
    if not os.path.isfile(path):
        log_status(f"File not found: {os.path.basename(path)}")
        return
    fname = os.path.basename(path)   # <-- only the file name

    try:
        img_local = sitk.ReadImage(path)
    except Exception as e:
        mri_load_label.value = _status_html("MRI", loaded=False)
        log_status(f"SimpleITK read failed: {e}")
        return

    try:
        vol = prepare_volume(img_local)
    except Exception as e:
        mri_load_label.value = _status_html("MRI", loaded=False)
        log_status(f"Normalization failed: {e}")
        return

    volume3d = vol
    image_sitk = img_local
    current_image_path = path
    mri_load_label.value = _status_html("MRI", loaded=True, name=fname)

    dim_z, dim_y, dim_x = volume3d.shape
    init_mask()

    axial_index = dim_z // 2
    coronal_index = dim_y // 2
    sagittal_index = dim_x // 2

    axial_center_x = dim_x / 2.0
    axial_center_y = dim_y / 2.0
    cor_center_x = dim_x / 2.0
    cor_center_z = dim_z / 2.0
    sag_center_y = dim_y / 2.0
    sag_center_z = dim_z / 2.0

    mip_angle = 0.0
    zoom_slider.value = zoom_slider_value_default

    axial_slider.min=0; axial_slider.max=dim_z-1; axial_slider.value=axial_index; axial_slider.disabled=False
    coronal_slider.min=0; coronal_slider.max=dim_y-1; coronal_slider.value=coronal_index; coronal_slider.disabled=False
    sagittal_slider.min=0; sagittal_slider.max=dim_x-1; sagittal_slider.value=sagittal_index; sagittal_slider.disabled=False

    box_prompts=[]; point_prompts=[]
    drawing=False; box_drawing=False
    current_view="axial"

    # reset SAM embedding + candidates
    last_sam_image_view = None
    last_sam_image_slice_idx = None
    sam_candidates = None
    candidate_thumbs_box.children = []

    stem, ext = split_name_and_ext(path)
    last_save_dir = os.path.dirname(path)
    save_filename_text.value = stem
    if ext in save_format_dropdown.options:
        save_format_dropdown.value = ext
    else:
        save_format_dropdown.value = '.nii.gz'
    save_dir_chooser.default_path = last_save_dir
    save_dir_chooser.reset()

    redraw()
    log_status(f"Loaded MRI: {fname} (Z={dim_z},Y={dim_y},X={dim_x}). View: Axial.")

# Auto-load MRI when user chooses a file
fc.register_callback(load_from_path)


############################################################
# Load SAMRI model (now auto-triggered from file chooser)
############################################################

def load_model(_chooser):
    global current_model_path, sam_model, sam_predictor, sam_device
    global last_sam_image_view, last_sam_image_slice_idx
    global sam_candidates

    path = fc_model.selected
    if not path:
        log_status("Please choose a SAMRI checkpoint file first.")
        return

    path = os.path.abspath(path)
    if not os.path.isfile(path):
        log_status(f"Model file not found: {os.path.basename(path)}")
        return
    fname = os.path.basename(path)   # <-- only the file name

    # Device selection: CUDA → MPS → CPU
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    try:
        model = load_sam_model(checkpoint=path, model_type="samri", device=device)
        predictor = SamPredictor(model)
    except Exception as e:
        model_load_label.value = _status_html("SAMRI", loaded=False)
        log_status(f"Failed to load SAMRI model: {e}")
        return


    current_model_path = path
    sam_model = model
    sam_predictor = predictor
    sam_device = device

    # Reset embedding cache & candidates
    last_sam_image_view = None
    last_sam_image_slice_idx = None
    sam_candidates = None
    candidate_thumbs_box.children = []
    model_load_label.value = _status_html("SAMRI", loaded=True, name=fname)

    log_status(f"SAMRI model loaded: {fname} on device '{device}'.")

# Auto-load model when user chooses a checkpoint
fc_model.register_callback(load_model)


############################################################
# Load MASK (auto-triggered from mask file chooser)
############################################################

def load_mask(_chooser):
    """
    Load a previously saved 3D mask file.
    - Requires an MRI to be loaded (so we know target shape).
    - Checks that mask shape matches volume3d.shape.
    - Converts to binary uint8 (0/1).
    """
    global mask3d, last_save_dir

    if volume3d is None:
        mask_load_label.value = _status_html("Mask", loaded=False)
        log_status("Please load an MRI before loading a mask.")
        return

    path = fc_mask.selected
    if not path:
        mask_load_label.value = _status_html("Mask", loaded=False)
        log_status("Please choose a mask file first.")
        return

    path = os.path.abspath(path)
    if not os.path.isfile(path):
        mask_load_label.value = _status_html("Mask", loaded=False)
        log_status(f"Mask file not found: {os.path.basename(path)}")
        return

    fname = os.path.basename(path)

    try:
        mask_img = sitk.ReadImage(path)
        mask_arr = sitk.GetArrayFromImage(mask_img)
    except Exception as e:
        mask_load_label.value = _status_html("Mask", loaded=False)
        log_status(f"Failed to read mask file: {e}")
        return

    # Normalise dimensions similar to prepare_volume
    if mask_arr.ndim == 2:
        # A 2D mask cannot represent full 3D volume here
        mask_load_label.value = _status_html("Mask", loaded=False)
        log_status(
            f"Loaded mask is 2D with shape {mask_arr.shape}. Expected 3D mask "
            f"with shape {volume3d.shape} (Z, Y, X)."
        )
        return
    elif mask_arr.ndim == 3:
        pass
    elif mask_arr.ndim == 4:
        mask_arr = mask_arr[0]
    else:
        mask_load_label.value = _status_html("Mask", loaded=False)
        log_status(f"Unsupported mask dimensions: {mask_arr.shape}")
        return

    if mask_arr.shape != volume3d.shape:
        mask_load_label.value = _status_html("Mask", loaded=False)
        log_status(
            f"Mask shape {mask_arr.shape} does not match MRI shape {volume3d.shape} (Z, Y, X)."
        )
        return

    # Convert to binary uint8 mask: >0 → 1, else 0
    mask_arr = (mask_arr > 0).astype(np.uint8)
    mask3d = mask_arr

    # Update save directory and file name hints to the location of this mask
    stem, ext = split_name_and_ext(path)
    last_save_dir = os.path.dirname(path)
    save_dir_chooser.default_path = last_save_dir
    save_dir_chooser.reset()
    if stem:
        save_filename_text.value = stem
    if ext in save_format_dropdown.options:
        save_format_dropdown.value = ext

    mask_load_label.value = _status_html("Mask", loaded=True, name=fname)
    redraw()
    log_status(f"Mask loaded from {fname} and applied to volume.")

# Auto-load mask when user chooses a mask file
fc_mask.register_callback(load_mask)


############################################################
# Clear / Save mask / Clear prompts
############################################################

def clear_prompts(_):
    global box_prompts, point_prompts, box_drawing, sam_candidates
    if volume3d is None:
        return
    box_prompts = []
    point_prompts = []
    box_drawing = False
    sam_candidates = None
    candidate_thumbs_box.children = []
    _draw_ui(current_view, get_current_slice())
    log_status("Cleared prompts and candidate masks.")

clear_prompt_button.on_click(clear_prompts)


def clear_mask(_):
    global sam_candidates
    if volume3d is None:
        return
    init_mask()
    sam_candidates = None
    candidate_thumbs_box.children = []
    redraw()
    log_status("Mask cleared.")

clear_mask_button.on_click(clear_mask)


def save_mask(_):
    global last_save_dir
    if volume3d is None or mask3d is None:
        log_status("Nothing to save (no volume or mask).")
        return

    dir_path = save_dir_chooser.selected or last_save_dir or os.getcwd()
    dir_path = os.path.abspath(dir_path)
    if not os.path.isdir(dir_path):
        log_status(f"Invalid save directory: {dir_path}")
        return

    stem = save_filename_text.value.strip()
    if not stem:
        stem = "mask3d"
    ext = save_format_dropdown.value
    filename = stem + ext
    save_path = os.path.join(dir_path, filename)

    # Post-processing for file format:
    #  - ensure binary 0/1 and uint8
    mask_arr = (mask3d > 0).astype(np.uint8)

    if ext == '.png':
        # Save ONLY the current view's slice as a 2D PNG
        if current_view == "axial":
            mask2d = mask_arr[axial_index]
        elif current_view == "coronal":
            mask2d = mask_arr[:, coronal_index, :]
        elif current_view == "sagittal":
            mask2d = mask_arr[:, :, sagittal_index]
        else:
            # fallback: axial mid-slice
            mask2d = mask_arr[axial_index]

        img2d = (mask2d * 255).astype(np.uint8)

        try:
            Image.fromarray(img2d).save(save_path)
        except Exception as e:
            log_status(f"Failed to save PNG mask: {e}")
            return
    else:
        # 3D medical formats via SimpleITK, with spatial metadata
        mask_img = sitk.GetImageFromArray(mask_arr)
        if image_sitk is not None:
            mask_img.CopyInformation(image_sitk)

        try:
            sitk.WriteImage(mask_img, save_path)
        except Exception as e:
            log_status(f"Failed to save mask: {e}")
            return


    last_save_dir = dir_path
    save_dir_chooser.default_path = last_save_dir
    save_dir_chooser.reset()
    log_status(f"Mask saved to {save_path}")

save_mask_button.on_click(save_mask)


############################################################
# SAMRI slice pre-processing & multi-mask Generate Mask
############################################################

def get_current_slice_2d():
    """Return current 2D slice as float32 (0–1) in its native orientation."""
    if current_view == "axial":
        return volume3d[axial_index]          # (Y, X)
    elif current_view == "coronal":
        return volume3d[:, coronal_index, :]  # (Z, X)
    elif current_view == "sagittal":
        return volume3d[:, :, sagittal_index] # (Z, Y)
    else:
        # For MIP, just reuse axial slice if needed (but we don't run SAM on MIP)
        return volume3d[axial_index]


def make_sam_input_from_slice(slice2d):
    """
    Follow inference-style preprocessing:
      - slice2d in [0,1] float
      - normalize to 0–255, uint8
      - expand to 3-channel RGB
    """
    img8 = (slice2d * 255.0).clip(0, 255).astype(np.uint8)
    img_rgb = np.stack([img8, img8, img8], axis=-1)  # (H, W, 3)
    return img_rgb


def write_mask2d_to_volume(view_name, slice_idx, mask2d):
    """Write a 2D mask into mask3d for given view/slice. Return True if ok."""
    if view_name == "axial":
        if mask2d.shape != (dim_y, dim_x):
            return False
        mask3d[slice_idx, :, :] = mask2d.astype(np.uint8)
        return True
    elif view_name == "coronal":
        if mask2d.shape != (dim_z, dim_x):
            return False
        mask3d[:, slice_idx, :] = mask2d.astype(np.uint8)
        return True
    elif view_name == "sagittal":
        if mask2d.shape != (dim_z, dim_y):
            return False
        mask3d[:, :, slice_idx] = mask2d.astype(np.uint8)
        return True
    return False


def apply_candidate_mask(idx):
    """Apply selected candidate mask to the current view/slice."""
    global sam_candidates

    if sam_candidates is None:
        log_status("No candidate masks available.")
        return

    view = sam_candidates['view']
    sl   = sam_candidates['slice_idx']
    masks = sam_candidates['masks']
    scores = sam_candidates['scores']

    if idx < 0 or idx >= masks.shape[0]:
        log_status("Invalid candidate index.")
        return

    # Ensure we are still on the same view/slice
    if current_view != view or get_current_slice() != sl:
        log_status("Candidate masks belong to a different slice/view. Regenerate on this slice to update.")
        return

    mask2d = (masks[idx] > 0).astype(np.uint8)

    ok = write_mask2d_to_volume(view, sl, mask2d)
    if not ok:
        log_status(f"Mask shape mismatch for view={view}. Got {mask2d.shape}.")
        return

    redraw()
    log_status(f"Applied candidate mask {idx} (score={scores[idx]:.3f}) on {view} slice {sl}.")


def make_mask_thumbnail(slice2d, mask2d, score, idx, thumb_size=64):
    """
    Create a small RGB thumbnail of the slice + mask overlay.
    Returns a VBox(Image, Button) where the button is clickable to apply mask idx.
    """
    # base grayscale
    img8 = (slice2d * 255.0).clip(0, 255).astype(np.uint8)
    base = Image.fromarray(img8).convert("RGB")

    # mask to overlay
    mask_arr = (mask2d > 0).astype(np.uint8)
    mask_img = Image.fromarray(mask_arr * 255).convert("L")

    # resize both to thumbnail size
    base = base.resize((thumb_size, thumb_size), Image.BILINEAR)
    mask_img = mask_img.resize((thumb_size, thumb_size), Image.NEAREST)

    base_np = np.array(base)
    mask_np = np.array(mask_img) > 0

    # overlay color = current overlay_color_hex
    r, g, b = hex_to_rgb(overlay_color_hex)
    overlay = base_np.copy()
    overlay[mask_np, 0] = r
    overlay[mask_np, 1] = g
    overlay[mask_np, 2] = b

    thumb = Image.fromarray(overlay.astype(np.uint8))

    # encode thumbnail to PNG bytes
    bio = BytesIO()
    thumb.save(bio, format='PNG')
    png_bytes = bio.getvalue()

    img_widget = WImage(
        value=png_bytes,
        format='png',
        width=thumb_size,
        height=thumb_size
    )

    # button that applies this candidate when clicked
    btn = Button(
        description=f"{idx}: {score:.3f}",
        layout={'width': f'{thumb_size + 20}px'}
    )

    def _on_click(_b, i=idx):
        apply_candidate_mask(i)

    btn.on_click(_on_click)

    return VBox([img_widget, btn])


def update_candidate_thumbnails(slice2d, masks, scores):
    """Update the tiny panel with thumbnails for each candidate mask."""
    if masks is None or masks.ndim != 3:
        candidate_thumbs_box.children = []
        return

    K = masks.shape[0]
    thumbs = []
    for i in range(K):
        thumbs.append(
            make_mask_thumbnail(slice2d, masks[i], float(scores[i]), i)
        )
    candidate_thumbs_box.children = thumbs


def on_generate_mask(_):
    global mask3d, last_sam_image_view, last_sam_image_slice_idx, sam_candidates

    if volume3d is None:
        log_status("No MRI loaded. Please load an image first.")
        return
    if sam_predictor is None:
        log_status("No SAMRI model loaded. Please load a checkpoint first.")
        return

    if current_view == "mip":
        log_status("Generate Mask works on Axial/Coronal/Sagittal slices, not MIP.")
        return

    slice_idx = get_current_slice()

    # 1) Prepare 2D slice image for SAM (uint8, H×W×3)
    slice2d = get_current_slice_2d()
    img_rgb = make_sam_input_from_slice(slice2d)

    # 2) Run predictor.set_image only if slice/view changed
    if not (last_sam_image_view == current_view and last_sam_image_slice_idx == slice_idx):
        try:
            sam_predictor.set_image(img_rgb)
        except Exception as e:
            log_status(f"SAMRI set_image failed: {e}")
            return
        last_sam_image_view = current_view
        last_sam_image_slice_idx = slice_idx

    H, W = img_rgb.shape[:2]

    # 3) Collect prompts for current view & slice (in original image coords)
    boxes_here = [b for b in box_prompts
                  if b['view'] == current_view and b['slice'] == slice_idx]
    points_here = [p for p in point_prompts
                   if p['view'] == current_view and p['slice'] == slice_idx]

    box_array = None
    if boxes_here:
        # Use the last box for now
        b = boxes_here[-1]
        x0 = float(b['ix0'])
        y0 = float(b['iy0'])
        x1 = float(b['ix1'])
        y1 = float(b['iy1'])
        x_min = max(0.0, min(x0, x1))
        y_min = max(0.0, min(y0, y1))
        x_max = min(float(W - 1), max(x0, x1))
        y_max = min(float(H - 1), max(y0, y1))
        box_array = np.array([[x_min, y_min, x_max, y_max]], dtype=np.float32)

    point_coords = None
    point_labels = None
    if points_here:
        # All points treated as positive for now (label=1)
        coords = []
        labels = []
        for p in points_here:
            ix = float(p['ix'])
            iy = float(p['iy'])
            if 0 <= ix < W and 0 <= iy < H:
                coords.append([ix, iy])
                labels.append(1)
        if coords:
            point_coords = np.array(coords, dtype=np.float32)
            point_labels = np.array(labels, dtype=np.int32)

    if box_array is None and point_coords is None:
        log_status("No prompts on this slice. Please draw a box or point first.")
        return

    # 4) Run SAMRI predictor with multimask_output=True
    try:
        masks, scores, logits = sam_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_array,
            multimask_output=True
        )
    except Exception as e:
        log_status(f"SAMRI predict failed: {e}")
        return

    if masks.ndim != 3:
        log_status(f"Unexpected masks shape: {masks.shape}")
        return

    # Store candidates
    sam_candidates = {
        'view': current_view,
        'slice_idx': slice_idx,
        'masks': masks,   # (K, H, W)
        'scores': scores  # (K,)
    }

    # update thumbnails panel
    update_candidate_thumbnails(slice2d, masks, scores)

    # pick best candidate and apply immediately
    K = masks.shape[0]
    best_idx = int(np.argmax(scores))

    apply_candidate_mask(best_idx)

    log_status(
        f"Generated {K} candidate masks on {current_view} slice {slice_idx}. "
        f"Best idx={best_idx} (score={scores[best_idx]:.3f}). "
        f"Click thumbnails to switch masks."
    )


generate_mask_button.on_click(on_generate_mask)


############################################################
# View switches
############################################################

def set_axial(_):
    global current_view, drawing, box_drawing
    current_view="axial"; drawing=False; box_drawing=False; redraw()

def set_coronal(_):
    global current_view, drawing, box_drawing
    current_view="coronal"; drawing=False; box_drawing=False; redraw()

def set_sagittal(_):
    global current_view, drawing, box_drawing
    current_view="sagittal"; drawing=False; box_drawing=False; redraw()

def set_mip(_):
    global current_view, drawing, box_drawing
    current_view="mip"; drawing=False; box_drawing=False; redraw()

axial_view_button.on_click(set_axial)
coronal_view_button.on_click(set_coronal)
sagittal_view_button.on_click(set_sagittal)
mip_view_button.on_click(set_mip)


############################################################
# Sliders & observers
############################################################

def on_axial_change(change):
    global axial_index
    axial_index=change['new']
    if current_view=="axial":
        draw_axial()

def on_coronal_change(change):
    global coronal_index
    coronal_index=change['new']
    if current_view=="coronal":
        draw_coronal()

def on_sag_change(change):
    global sagittal_index
    sagittal_index=change['new']
    if current_view=="sagittal":
        draw_sagittal()

axial_slider.observe(on_axial_change, names='value')
coronal_slider.observe(on_coronal_change, names='value')
sagittal_slider.observe(on_sag_change, names='value')

# Axial +/- logic
def axial_minus(_):
    if axial_slider.disabled: 
        return
    axial_slider.value = max(axial_slider.min, axial_slider.value - 1)

def axial_plus(_):
    if axial_slider.disabled:
        return
    axial_slider.value = min(axial_slider.max, axial_slider.value + 1)

axial_minus_btn.on_click(axial_minus)
axial_plus_btn.on_click(axial_plus)


# Coronal +/- logic
def coronal_minus(_):
    if coronal_slider.disabled:
        return
    coronal_slider.value = max(coronal_slider.min, coronal_slider.value - 1)

def coronal_plus(_):
    if coronal_slider.disabled:
        return
    coronal_slider.value = min(coronal_slider.max, coronal_slider.value + 1)

coronal_minus_btn.on_click(coronal_minus)
coronal_plus_btn.on_click(coronal_plus)


# Sagittal +/- logic
def sagittal_minus(_):
    if sagittal_slider.disabled:
        return
    sagittal_slider.value = max(sagittal_slider.min, sagittal_slider.value - 1)

def sagittal_plus(_):
    if sagittal_slider.disabled:
        return
    sagittal_slider.value = min(sagittal_slider.max, sagittal_slider.value + 1)

sagittal_minus_btn.on_click(sagittal_minus)
sagittal_plus_btn.on_click(sagittal_plus)


def on_color_change(_):
    if volume3d is not None:
        redraw()

color_picker.observe(on_color_change, names='value')

def on_show_mask_change(_):
    if volume3d is not None:
        redraw()

show_mask_checkbox.observe(on_show_mask_change, names='value')

def on_zoom_change(_):
    if volume3d is not None:
        redraw()

zoom_slider.observe(on_zoom_change, names='value')

def on_canvas_size_change(change):
    new_size = int(change['new'])
    if new_size <= 0:
        return
    main_canv.layout.width = f"{new_size}px"
    main_canv.layout.height = f"{new_size}px"
    log_status(f"Canvas display size set to {new_size} x {new_size}px")

canvas_size_slider.observe(on_canvas_size_change, names='value')


############################################################
# Zoom +/- buttons
############################################################

def zoom_out(_):
    zoom_slider.value = max(zoom_slider.min, zoom_slider.value - 10)

def zoom_in(_):
    zoom_slider.value = min(zoom_slider.max, zoom_slider.value + 10)

zoom_out_button.on_click(zoom_out)
zoom_in_button.on_click(zoom_in)


############################################################
# Canvas size +/- buttons
############################################################

def canvas_size_minus(_):
    canvas_size_slider.value = max(canvas_size_slider.min,
                                   canvas_size_slider.value - canvas_size_slider.step)

def canvas_size_plus(_):
    canvas_size_slider.value = min(canvas_size_slider.max,
                                   canvas_size_slider.value + canvas_size_slider.step)

canvas_size_minus_button.on_click(canvas_size_minus)
canvas_size_plus_button.on_click(canvas_size_plus)


############################################################
# Layout
############################################################

controls_tools = HBox([
    tool_selector,
    brush_size_slider,
    color_picker,
    show_mask_checkbox
])

# Put clear mask + generate + thumbnails on same logical section
controls_prompts = HBox([
    clear_prompt_button,
    clear_mask_button,
    generate_mask_button
])

controls_slices = HBox([
    HBox([axial_minus_btn, axial_slider, axial_plus_btn]),
    HBox([coronal_minus_btn, coronal_slider, coronal_plus_btn]),
    HBox([sagittal_minus_btn, sagittal_slider, sagittal_plus_btn])
])

controls_view     = HBox([axial_view_button, coronal_view_button, sagittal_view_button, mip_view_button])

controls_actions  = HBox([save_mask_button])

save_controls = VBox([
    save_dir_chooser,
    HBox([save_filename_text, save_format_dropdown])
])

file_choosers_row = HBox([
    VBox([fc, mri_load_label]),
    VBox([fc_model, model_load_label]),
    VBox([fc_mask, mask_load_label])
])


canvas_size_controls = HBox([
    canvas_size_slider,
    canvas_size_minus_button,
    canvas_size_plus_button,
    zoom_slider,
    zoom_out_button,
    zoom_in_button
])

ui = VBox([
    status_label,
    file_choosers_row,
    controls_tools,
    canvas_size_controls,
    controls_prompts,
    candidate_thumbs_box,   # tiny panel with thumbnails
    controls_slices,
    controls_view,
    VBox([
        Label("Current view"),
        main_canv
    ]),
    Label("Save mask:"),
    save_controls,
    controls_actions
])

ui
