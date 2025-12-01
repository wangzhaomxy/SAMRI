############################################################
# MRI Viewer with Mask Editing, Zoom/Pan, 3D MIP rotate,
# Canvas display size, Save Options and SAMRI model chooser
############################################################

import os
import math
import time

import numpy as np
import SimpleITK as sitk
from PIL import Image

from ipycanvas import MultiCanvas
from ipywidgets import (
    VBox, HBox, Button, ColorPicker, IntSlider,
    ToggleButtons, Checkbox, Label, Text, Dropdown
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

box_prompts = []
point_prompts = []

overlay_color_hex = "#ff0000"

# For preserving metadata and save defaults
image_sitk = None          # last loaded SimpleITK image
current_image_path = None  # path of last loaded image
last_save_dir = None       # last folder used to save mask

# Placeholder for model path (SAMRI checkpoint)
current_model_path = None

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

# Initial display size matches internal resolution
main_canv.layout.width = f"{VIEW_SIZE}px"
main_canv.layout.height = f"{VIEW_SIZE}px"

############################################################
# Widgets
############################################################

status_label = Label(
    value="Choose an MRI file → Load Image → View appears. Default view: Axial mid-slice."
)

# --- File chooser for loading MRI ---
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

# renamed: Load Image
load_path_button = Button(
    description='Load Image',
    button_style=''
)

load_model_button = Button(
    description='Load model',
    button_style=''
)

tool_selector = ToggleButtons(
    options=['Brush', 'Box', 'Point', 'Hand'],
    value='Brush',
    description='Tool:'
)

mode_selector = ToggleButtons(
    options=['Paint', 'Erase'],
    value='Paint',
    description='Mode:'
)

# default brush size = 1
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

# Zoom slider 0–400%, default 100%
zoom_slider = IntSlider(
    value=zoom_slider_value_default,
    min=0,
    max=400,
    step=10,
    description='Zoom (%):'
)

# Zoom - / +
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

# View switching buttons
axial_view_button = Button(description="Axial")
coronal_view_button = Button(description="Coronal")
sagittal_view_button = Button(description="Sagittal")
mip_view_button = Button(description="3D MIP")

# --- Canvas display size control (CSS only) ---
canvas_size_slider = IntSlider(
    value=VIEW_SIZE,
    min=128,
    max=512,
    step=64,
    description='Canvas size:'
)

# --- Save mask controls ---
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
    options=['.nii.gz', '.mhd', '.mha', '.nrrd', '.dcm'],
    value='.nii.gz',
    description='Format:'
)


############################################################
# Helper functions
############################################################

def log_status(msg):
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
    for ext in ['.nii', '.mhd', '.mha', '.nrrd', '.dcm', '.gz']:
        if base.lower().endswith(ext):
            return base[:-len(ext)], ext
    return base, ''


def get_view_window(view_name):
    """
    For zoom >= 1: compute crop window in slice coordinates (left, top, w_view, h_view).
    For zoom < 1: return full slice and reset centers to mid; shrinking & padding handled
    in drawing & painting functions.
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

    # For zoom <= 1: use full slice (no cropping), center reset.
    if zf <= 1.0:
        if view_name == "axial":
            axial_center_x, axial_center_y = w / 2.0, h / 2.0
        elif view_name == "coronal":
            cor_center_x, cor_center_z = w / 2.0, h / 2.0
        elif view_name == "sagittal":
            sag_center_y, sag_center_z = w / 2.0, h / 2.0
        return 0.0, 0.0, float(w), float(h)

    # For zoom > 1: crop window smaller than full slice
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
# Drawing slices (zoom < 1 uses shrink+pad)
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
    Common helper for axial/coronal/sagittal:
    - If zoom <= 1: shrink full slice to VIEW_SIZE * zoom, center with black padding.
    - If zoom > 1: use cropping window from get_view_window, fill whole canvas.
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

        # Base black background
        rgba = np.zeros((VIEW_SIZE, VIEW_SIZE, 4), dtype=np.uint8)
        rgba[..., 3] = 255  # opaque black background

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
        # zoom > 1: crop & fill canvas
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
    slice2d = volume3d[axial_index]  # (Y, X)
    mask_slice = mask3d[axial_index] if (mask3d is not None) else None
    _draw_slice_zoomed(slice2d, mask_slice, "axial", axial_index)


def draw_coronal():
    if volume3d is None:
        main_bg.clear(); main_ol.clear(); main_ui.clear()
        return
    slice2d = volume3d[:, coronal_index, :]  # (Z, X)
    mask_slice = mask3d[:, coronal_index, :] if (mask3d is not None) else None
    _draw_slice_zoomed(slice2d, mask_slice, "coronal", coronal_index)


def draw_sagittal():
    if volume3d is None:
        main_bg.clear(); main_ol.clear(); main_ui.clear()
        return
    slice2d = volume3d[:, :, sagittal_index]  # (Z, Y)
    mask_slice = mask3d[:, :, sagittal_index] if (mask3d is not None) else None
    _draw_slice_zoomed(slice2d, mask_slice, "sagittal", sagittal_index)


def draw_mip():
    main_bg.clear(); main_ol.clear(); main_ui.clear()
    if volume3d is None:
        return

    global mip_angle

    mip = volume3d.max(axis=0)  # (Y, X)
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
        mip_mask = (mask3d > 0).max(axis=0)  # (Y, X)
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
# Painting (zoom-aware, with black border when zoom < 1)
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
        return vz, vx  # (z, x)
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
        return vz, vy  # (z, y)
    else:
        left, top, width_view, height_view = get_view_window("sagittal")
        vy = left + (cx / VIEW_SIZE) * width_view
        vz = top + (cy / VIEW_SIZE) * height_view
        vy = int(np.clip(round(vy), 0, dim_y - 1))
        vz = int(np.clip(round(vz), 0, dim_z - 1))
        return vz, vy


def paint_circle(cx, cy):
    if mask3d is None:
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

        if mode_selector.value=="Paint":
            mask3d[z,y0:y1,x0:x1][region]=1
        else:
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

        if mode_selector.value=="Paint":
            mask3d[z0:z1,y,x0:x1][region]=1
        else:
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

        if mode_selector.value=="Paint":
            mask3d[z0:z1,y0:y1,x][region]=1
        else:
            mask3d[z0:z1,y0:y1,x][region]=0

    redraw()


def paint_line(x0,y0,x1,y1):
    steps = int(max(abs(x1-x0),abs(y1-y0))/max(brush_size_slider.value/2,1))+1
    for t in np.linspace(0,1,steps):
        paint_circle(x0+t*(x1-x0), y0+t*(y1-y0))


############################################################
# Panning for Hand tool (disabled when zoom <= 1)
############################################################

def pan_view(view_name, dx_canvas, dy_canvas):
    global axial_center_x, axial_center_y
    global cor_center_x, cor_center_z
    global sag_center_y, sag_center_z

    # No pan when zoom <= 1 (full slice in view)
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

def on_mouse_down(x,y):
    global drawing, last_canvas_x, last_canvas_y, box_drawing

    if volume3d is None:
        return

    tool = tool_selector.value

    if current_view=="mip" and tool in ("Brush", "Box", "Point"):
        return

    if tool=="Brush":
        drawing=True
        last_canvas_x, last_canvas_y = x,y
        paint_circle(x,y)

    elif tool=="Box":
        if current_view == "mip":
            return
        slice_idx = get_current_slice()
        box_prompts.append({
            'view': current_view,
            'slice': slice_idx,
            'x0':x, 'y0':y, 'x1':x, 'y1':y
        })
        box_drawing=True
        _draw_ui(current_view, slice_idx)

    elif tool=="Point":
        if current_view == "mip":
            return
        slice_idx=get_current_slice()
        point_prompts.append({
            'view':current_view,
            'slice':slice_idx,
            'x':x,'y':y
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

    if tool=="Brush" and drawing and current_view != "mip":
        paint_line(last_canvas_x, last_canvas_y, x, y)
        last_canvas_x, last_canvas_y = x,y

    elif tool=="Box" and box_drawing and current_view != "mip":
        if box_prompts:
            box_prompts[-1]['x1']=x
            box_prompts[-1]['y1']=y
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

    if tool=="Brush":
        drawing=False

    elif tool=="Box" and box_drawing:
        if current_view != "mip":
            box_prompts[-1]['x1']=x
            box_prompts[-1]['y1']=y
            _draw_ui(current_view, get_current_slice())
        box_drawing=False

    elif tool=="Hand":
        drawing=False


main_canv.on_mouse_down(on_mouse_down)
main_canv.on_mouse_move(on_mouse_move)
main_canv.on_mouse_up(on_mouse_up)


############################################################
# Load MRI button
############################################################

def load_from_path(_):
    global volume3d, mask3d
    global dim_z, dim_y, dim_x
    global axial_index, coronal_index, sagittal_index
    global box_prompts, point_prompts, drawing, box_drawing, current_view
    global image_sitk, current_image_path, last_save_dir
    global axial_center_x, axial_center_y
    global cor_center_x, cor_center_z
    global sag_center_y, sag_center_z
    global mip_angle

    path = fc.selected
    if not path:
        log_status("Please choose an MRI file first.")
        return

    path = os.path.abspath(path)
    if not os.path.isfile(path):
        log_status(f"File not found: {path}")
        return

    try:
        img_local = sitk.ReadImage(path)
    except Exception as e:
        log_status(f"SimpleITK read failed: {e}")
        return

    try:
        vol = prepare_volume(img_local)
    except Exception as e:
        log_status(f"Normalization failed: {e}")
        return

    volume3d = vol
    image_sitk = img_local
    current_image_path = path

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
    log_status(f"Loaded MRI: {path} (Z={dim_z},Y={dim_y},X={dim_x}). View: Axial.")


load_path_button.on_click(load_from_path)


############################################################
# Load model button (placeholder)
############################################################

def load_model(_):
    global current_model_path
    path = fc_model.selected
    if not path:
        log_status("Please choose a SAMRI checkpoint file first.")
        return

    path = os.path.abspath(path)
    if not os.path.isfile(path):
        log_status(f"Model file not found: {path}")
        return

    current_model_path = path
    log_status(f"[Placeholder] SAMRI checkpoint selected: {path}. Model loading not implemented yet.")

load_model_button.on_click(load_model)


############################################################
# Clear / Save mask / Clear prompts
############################################################

def clear_prompts(_):
    global box_prompts, point_prompts, box_drawing
    if volume3d is None: return
    box_prompts=[]; point_prompts=[]; box_drawing=False
    _draw_ui(current_view, get_current_slice())
    log_status("Cleared prompts.")

clear_prompt_button.on_click(clear_prompts)


def clear_mask(_):
    if volume3d is None: return
    init_mask()
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

    mask_img = sitk.GetImageFromArray(mask3d.astype(np.uint8))
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
# Sliders & other observers
############################################################

def get_current_slice():
    if current_view == "axial":
        return axial_index
    if current_view == "coronal":
        return coronal_index
    if current_view == "sagittal":
        return sagittal_index
    return axial_index

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
# Zoom + / Zoom - buttons
############################################################

def zoom_out(_):
    zoom_slider.value = max(zoom_slider.min, zoom_slider.value - 10)

def zoom_in(_):
    zoom_slider.value = min(zoom_slider.max, zoom_slider.value + 10)

zoom_out_button.on_click(zoom_out)
zoom_in_button.on_click(zoom_in)


############################################################
# Layout
############################################################

controls_tools = HBox([
    tool_selector, mode_selector, brush_size_slider,
    color_picker, show_mask_checkbox
])

# Clear mask next to Clear prompts
controls_prompts = HBox([clear_prompt_button, clear_mask_button])

# Slice sliders + zoom slider + +/- buttons
controls_slices   = HBox([
    axial_slider,
    coronal_slider,
    sagittal_slider,
    zoom_slider,
    zoom_out_button,
    zoom_in_button
])

# View buttons BETWEEN sliders and canvas
controls_view     = HBox([axial_view_button, coronal_view_button, sagittal_view_button, mip_view_button])

controls_actions  = HBox([save_mask_button])

save_controls = VBox([
    save_dir_chooser,
    HBox([save_filename_text, save_format_dropdown])
])

file_choosers_row = HBox([fc, fc_model])
load_buttons_row = HBox([load_path_button, load_model_button])

ui = VBox([
    status_label,
    file_choosers_row,
    load_buttons_row,
    controls_tools,
    canvas_size_slider,   # canvas display size widget
    controls_prompts,
    controls_slices,      # sliders + zoom
    controls_view,        # view buttons
    VBox([
        Label("Current view"),
        main_canv
    ]),
    Label("Save mask:"),
    save_controls,
    controls_actions
])

ui
