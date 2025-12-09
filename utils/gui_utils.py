############################################################
# MRI Viewer + SAMRI inference integration (slice-level)
# - Multi-view MRI with Paint/Erase/Box/Point/Hand tools
# - SAMRI checkpoint loading (load_sam_model + SamPredictor)
# - Generate Mask button with multi-mask selection via thumbnails
# - Multi-label masks with per-label colors, names & visibility
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
    VBox, HBox, Button, ColorPicker, IntSlider, BoundedIntText,
    ToggleButtons, Checkbox, Label, Text, Dropdown,
    Image as WImage, HTML
)
from ipyfilechooser import FileChooser

VIEW_SIZE = 256

DEFAULT_LABEL_PALETTE = [
    "#ff0000", "#00ff00", "#0000ff", "#ffff00",
    "#ff00ff", "#00ffff", "#ffa500", "#800080",
    "#008000", "#000080"
]


class MRIViewerApp:
    def __init__(self):
        # Volume / mask
        self.volume3d = None         # float32 (Z, Y, X)
        self.mask3d = None           # uint8 (Z, Y, X), multi-label
        self.dim_z = self.dim_y = self.dim_x = 0

        # Slice indices
        self.axial_index = 0
        self.coronal_index = 0
        self.sagittal_index = 0
        self.current_view = "axial"

        # Drawing state
        self.drawing = False
        self.last_canvas_x = None
        self.last_canvas_y = None
        self.box_drawing = False

        # Prompts
        self.box_prompts = []
        self.point_prompts = []

        # Labels
        self.label_colors = {1: DEFAULT_LABEL_PALETTE[0]}  # label_id -> color
        self.current_label = 1
        self.label_visibility = {}   # label_id -> bool
        self.label_names = {}        # label_id -> name str
        self.label_checkboxes = {}   # label_id -> checkbox widget
        self.legend_batch_update = False

        # MRI / mask / model paths
        self.image_sitk = None
        self.current_image_path = None
        self.last_save_dir = None

        self.current_model_path = None
        self.sam_model = None
        self.sam_predictor = None
        self.sam_device = None

        # SAM embedding & candidates
        self.last_sam_image_view = None
        self.last_sam_image_slice_idx = None
        self.sam_candidates = None   # {view, slice_idx, masks, scores, label_id, color_hex}

        # Zoom / pan / MIP
        self.zoom_slider_value_default = 100
        self.axial_center_x = self.axial_center_y = 0.0
        self.cor_center_x = self.cor_center_z = 0.0
        self.sag_center_y = self.sag_center_z = 0.0
        self.mip_angle = 0.0

        # Build UI & wiring
        self._build_canvas()
        self._build_widgets()
        self._wire_events()

        # Initial label legend
        self.update_label_state_from_mask()

    # -----------------------------------------------------
    # Utility helpers
    # -----------------------------------------------------
    @staticmethod
    def _status_html(kind: str, loaded: bool, name: str = "") -> str:
        color = "green" if loaded else "red"
        state = "loaded" if loaded else "unloaded"
        extra = f" ({name})" if (loaded and name) else ""
        return f"<span style='color:{color}; font-weight:bold;'>{kind}: {state}{extra}</span>"

    def log_status(self, msg: str):
        self.status_label.value = msg

    @staticmethod
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def get_zoom_factor(self):
        z = self.zoom_slider.value / 100.0
        return max(z, 0.01)

    # -----------------------------------------------------
    # Label utilities & legend
    # -----------------------------------------------------
    def get_label_color(self, label: int) -> str:
        if label <= 0:
            return "#000000"
        if label not in self.label_colors:
            idx = (len(self.label_colors)) % len(DEFAULT_LABEL_PALETTE)
            self.label_colors[label] = DEFAULT_LABEL_PALETTE[idx]
        return self.label_colors[label]

    def make_label_legend_row(self, label_id: int):
        color_hex = self.get_label_color(label_id)
        visible = self.label_visibility.get(label_id, True)
        self.label_visibility[label_id] = visible

        cb = Checkbox(
            value=visible,
            description='',
            indent=False,
            layout={'width': '24px'}
        )
        self.label_checkboxes[label_id] = cb

        color_box = HTML(
            value=(
                f"<div style='width:16px; height:16px; "
                f"background:{color_hex}; border:1px solid #000;'></div>"
            )
        )

        id_label = Label(f"ID {label_id}")

        name_str = self.label_names.get(label_id, "")
        name_text = Text(
            value=name_str,
            placeholder=f"Label {label_id}",
            layout={'width': '140px'}
        )

        def _on_cb_change(change, lab=label_id):
            if change['name'] != 'value':
                return
            self.label_visibility[lab] = bool(change['new'])
            if self.legend_batch_update:
                return

            labels = [l for l in self.label_visibility.keys()]
            if labels:
                all_on = all(self.label_visibility.get(l, True) for l in labels)
            else:
                all_on = True

            self.legend_batch_update = True
            self.all_labels_checkbox.value = all_on
            self.legend_batch_update = False

            if self.volume3d is not None:
                self.redraw()

        cb.observe(_on_cb_change, names='value')

        def _on_name_change(change, lab=label_id):
            if change['name'] != 'value':
                return
            self.label_names[lab] = change['new']

        name_text.observe(_on_name_change, names='value')

        return HBox([cb, color_box, id_label, name_text])

    def update_label_legend(self):
        self.label_checkboxes.clear()

        labels = set(self.label_colors.keys())
        if self.mask3d is not None:
            labels.update(np.unique(self.mask3d).tolist())
        labels.discard(0)
        if not labels:
            labels = {1}

        rows = [self.make_label_legend_row(int(lab)) for lab in sorted(labels)]
        self.label_legend_rows_box.children = rows

        labels_list = list(self.label_visibility.keys())
        if labels_list:
            all_on = all(self.label_visibility.get(l, True) for l in labels_list)
        else:
            all_on = True

        self.legend_batch_update = True
        self.all_labels_checkbox.value = all_on
        self.legend_batch_update = False

    def update_label_state_from_mask(self):
        labels = set()
        if self.mask3d is not None:
            labels.update(np.unique(self.mask3d).tolist())
        labels.discard(0)
        if not labels:
            labels = {1}
        for lab in sorted(labels):
            self.get_label_color(int(lab))

        if self.current_label not in labels:
            self.current_label = sorted(labels)[0]

        self.current_label_input.value = self.current_label
        self.label_color_picker.value = self.get_label_color(self.current_label)

        self.update_label_legend()

    # -----------------------------------------------------
    # Volume / mask init
    # -----------------------------------------------------
    @staticmethod
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

    def init_mask(self):
        if self.volume3d is None:
            self.mask3d = None
        else:
            self.mask3d = np.zeros_like(self.volume3d, dtype=np.uint8)

    @staticmethod
    def split_name_and_ext(path):
        base = os.path.basename(path)
        if base.lower().endswith('.nii.gz'):
            return base[:-7], '.nii.gz'
        for ext in ['.nii', '.mhd', '.mha', '.nrrd', '.dcm', '.gz', '.png']:
            if base.lower().endswith(ext):
                return base[:-len(ext)], ext
        return base, ''

    # -----------------------------------------------------
    # Canvas & widgets
    # -----------------------------------------------------
    def _build_canvas(self):
        self.main_canv = MultiCanvas(3, width=VIEW_SIZE, height=VIEW_SIZE)
        self.main_bg = self.main_canv[0]
        self.main_ol = self.main_canv[1]
        self.main_ui = self.main_canv[2]
        self.main_canv.layout.width = f"{VIEW_SIZE}px"
        self.main_canv.layout.height = f"{VIEW_SIZE}px"

    def _build_widgets(self):
        self.status_label = Label(
            value="Choose an MRI file → image loads automatically → view appears. "
                  "Then choose a SAMRI checkpoint → model loads automatically."
        )

        # MRI chooser
        self.fc = FileChooser(os.getcwd(), select_default=True, show_only_dirs=False)
        self.fc.title = "<b>Choose MRI file</b>"
        self.fc.filter_pattern = [
            "*.nii", "*.nii.gz",
            "*.mha", "*.mhd",
            "*.nrrd",
            "*.dcm",
            "*.png", "*.jpg", "*.jpeg",
            "*.gz"
        ]

        # Model chooser
        self.fc_model = FileChooser(os.getcwd(), select_default=True, show_only_dirs=False)
        self.fc_model.title = "<b>Choose SAMRI checkpoint</b>"
        self.fc_model.filter_pattern = [
            "*.pt", "*.pth", "*.ckpt", "*.onnx", "*.bin", "*.safetensors"
        ]

        # Mask chooser
        self.fc_mask = FileChooser(os.getcwd(), select_default=True, show_only_dirs=False)
        self.fc_mask.title = "<b>Choose mask file</b>"
        self.fc_mask.filter_pattern = [
            "*.nii", "*.nii.gz",
            "*.mhd", "*.mha",
            "*.nrrd",
            "*.dcm"
        ]

        # Tools
        self.tool_selector = ToggleButtons(
            options=['Hand', 'Paint', 'Erase', 'Box', 'Point'],
            value='Hand',
            description='Tool:'
        )

        self.brush_size_slider = BoundedIntText(
            value=1,
            min=1,
            max=40,
            step=1,
            description='Brush size:'
        )

        # Labels (current id + color + add/delete)
        self.current_label_input = BoundedIntText(
            value=self.current_label,
            min=0,
            max=9999,
            step=1,
            description='Label ID:'
        )

        self.add_label_button = Button(
            description='Add new label',
            button_style='info'
        )

        self.delete_label_button = Button(
            description='Delete',
            button_style='warning'
        )

        self.label_color_picker = ColorPicker(
            concise=False,
            description='Label color:',
            value=self.label_colors[self.current_label]
        )

        # Slices
        self.axial_slider = IntSlider(
            value=0, min=0, max=0, step=1,
            description='Axial (Z):', disabled=True
        )
        self.coronal_slider = IntSlider(
            value=0, min=0, max=0, step=1,
            description='Coronal (Y):', disabled=True
        )
        self.sagittal_slider = IntSlider(
            value=0, min=0, max=0, step=1,
            description='Sagittal (X):', disabled=True
        )

        self.axial_minus_btn = Button(description="-", layout={'width': '40px'})
        self.axial_plus_btn  = Button(description="+", layout={'width': '40px'})
        self.coronal_minus_btn = Button(description="-", layout={'width': '40px'})
        self.coronal_plus_btn  = Button(description="+", layout={'width': '40px'})
        self.sagittal_minus_btn = Button(description="-", layout={'width': '40px'})
        self.sagittal_plus_btn  = Button(description="+", layout={'width': '40px'})

        self.zoom_slider = IntSlider(
            value=self.zoom_slider_value_default,
            min=0, max=400, step=10,
            description='Zoom (%):'
        )
        self.zoom_out_button = Button(description='Zoom -')
        self.zoom_in_button  = Button(description='Zoom +')

        self.clear_prompt_button = Button(
            description='Clear prompts',
            button_style='danger'
        )
        self.clear_mask_button = Button(
            description='Clear mask',
            button_style='warning'
        )
        self.save_mask_button = Button(
            description='Save mask',
            button_style='success'
        )
        self.generate_mask_button = Button(
            description='Generate Mask',
            button_style='info'
        )

        self.candidate_thumbs_box = HBox([])

        self.axial_view_button = Button(description="Axial")
        self.coronal_view_button = Button(description="Coronal")
        self.sagittal_view_button = Button(description="Sagittal")
        self.mip_view_button = Button(description="3D MIP")

        self.canvas_size_slider = IntSlider(
            value=VIEW_SIZE,
            min=128,
            max=512,
            step=64,
            description='Canvas size:'
        )
        self.canvas_size_minus_button = Button(description='Size -')
        self.canvas_size_plus_button  = Button(description='Size +')

        self.save_dir_chooser = FileChooser(
            os.getcwd(),
            select_default=True,
            show_only_dirs=True
        )
        self.save_dir_chooser.title = "<b>Choose folder to save mask</b>"

        self.save_filename_text = Text(
            value='',
            description='Name:',
            placeholder='mask3d'
        )

        self.save_format_dropdown = Dropdown(
            options=['.nii.gz', '.mhd', '.mha', '.nrrd', '.dcm', '.png'],
            value='.nii.gz',
            description='Format:'
        )

        self.mri_load_label   = HTML(value=self._status_html("MRI",   False))
        self.model_load_label = HTML(value=self._status_html("SAMRI", False))
        self.mask_load_label  = HTML(value=self._status_html("Mask",  False))

        # Legend
        self.all_labels_checkbox = Checkbox(
            value=True,
            description='ALL',
            indent=False
        )
        self.label_legend_rows_box = VBox([])
        self.label_legend_box = VBox([self.all_labels_checkbox, self.label_legend_rows_box])

        # Layout
        tools_left = HBox([self.tool_selector, self.brush_size_slider])
        label_block = VBox([
            HBox([self.current_label_input, self.label_color_picker,
                  self.add_label_button, self.delete_label_button])
        ])
        self.controls_tools = HBox([tools_left, label_block])

        self.controls_prompts = HBox([
            self.clear_prompt_button,
            self.clear_mask_button,
            self.generate_mask_button
        ])

        self.controls_slices = HBox([
            HBox([self.axial_minus_btn, self.axial_slider, self.axial_plus_btn]),
            HBox([self.coronal_minus_btn, self.coronal_slider, self.coronal_plus_btn]),
            HBox([self.sagittal_minus_btn, self.sagittal_slider, self.sagittal_plus_btn])
        ])

        self.controls_view = HBox([
            self.axial_view_button, self.coronal_view_button,
            self.sagittal_view_button, self.mip_view_button
        ])

        self.controls_actions = HBox([self.save_mask_button])

        self.save_controls = VBox([
            self.save_dir_chooser,
            HBox([self.save_filename_text, self.save_format_dropdown])
        ])

        self.file_choosers_row = HBox([
            VBox([self.fc, self.mri_load_label]),
            VBox([self.fc_model, self.model_load_label]),
            VBox([self.fc_mask, self.mask_load_label])
        ])

        self.canvas_size_controls = HBox([
            self.canvas_size_slider, self.canvas_size_minus_button, self.canvas_size_plus_button,
            self.zoom_slider, self.zoom_out_button, self.zoom_in_button
        ])

        self.ui = VBox([
            self.status_label,
            self.file_choosers_row,
            self.controls_tools,
            self.canvas_size_controls,
            self.controls_prompts,
            self.candidate_thumbs_box,
            self.controls_slices,
            self.controls_view,
            HBox([
                VBox([
                    Label("Current view"),
                    self.main_canv
                ]),
                VBox([
                    Label("Label legend & visibility"),
                    self.label_legend_box
                ])
            ]),
            Label("Save mask:"),
            self.save_controls,
            self.controls_actions
        ])

    # -----------------------------------------------------
    # Event wiring
    # -----------------------------------------------------
    def _wire_events(self):
        # File choosers
        self.fc.register_callback(self.load_from_path)
        self.fc_model.register_callback(self.load_model)
        self.fc_mask.register_callback(self.load_mask)

        # Canvas mouse
        self.main_canv.on_mouse_down(self.on_mouse_down)
        self.main_canv.on_mouse_move(self.on_mouse_move)
        self.main_canv.on_mouse_up(self.on_mouse_up)

        # Slice sliders
        self.axial_slider.observe(self.on_axial_change, names='value')
        self.coronal_slider.observe(self.on_coronal_change, names='value')
        self.sagittal_slider.observe(self.on_sag_change, names='value')

        self.axial_minus_btn.on_click(self.axial_minus)
        self.axial_plus_btn.on_click(self.axial_plus)
        self.coronal_minus_btn.on_click(self.coronal_minus)
        self.coronal_plus_btn.on_click(self.coronal_plus)
        self.sagittal_minus_btn.on_click(self.sagittal_minus)
        self.sagittal_plus_btn.on_click(self.sagittal_plus)

        # Views
        self.axial_view_button.on_click(lambda _: self.set_view("axial"))
        self.coronal_view_button.on_click(lambda _: self.set_view("coronal"))
        self.sagittal_view_button.on_click(lambda _: self.set_view("sagittal"))
        self.mip_view_button.on_click(lambda _: self.set_view("mip"))

        # Zoom / size
        self.zoom_slider.observe(self.on_zoom_change, names='value')
        self.zoom_out_button.on_click(self.zoom_out)
        self.zoom_in_button.on_click(self.zoom_in)
        self.canvas_size_slider.observe(self.on_canvas_size_change, names='value')
        self.canvas_size_minus_button.on_click(self.canvas_size_minus)
        self.canvas_size_plus_button.on_click(self.canvas_size_plus)

        # Label controls
        self.label_color_picker.observe(self.on_label_color_change, names='value')
        self.current_label_input.observe(self.on_current_label_input_change, names='value')
        self.add_label_button.on_click(self.on_add_label_clicked)
        self.delete_label_button.on_click(self.on_delete_label_clicked)
        self.all_labels_checkbox.observe(self.on_all_labels_change, names='value')

        # Prompts/mask
        self.clear_prompt_button.on_click(self.clear_prompts)
        self.clear_mask_button.on_click(self.clear_mask)
        self.save_mask_button.on_click(self.save_mask)
        self.generate_mask_button.on_click(self.on_generate_mask)

    # -----------------------------------------------------
    # View / zoom helpers
    # -----------------------------------------------------
    def set_view(self, view_name: str):
        self.current_view = view_name
        self.drawing = False
        self.box_drawing = False
        self.redraw()

    def get_view_window(self, view_name):
        zf = self.get_zoom_factor()

        if view_name == "axial":
            h, w = self.dim_y, self.dim_x
            cx, cy = self.axial_center_x, self.axial_center_y
        elif view_name == "coronal":
            h, w = self.dim_z, self.dim_x
            cx, cy = self.cor_center_x, self.cor_center_z
        elif view_name == "sagittal":
            h, w = self.dim_z, self.dim_y
            cx, cy = self.sag_center_y, self.sag_center_z
        else:
            return 0.0, 0.0, 0.0, 0.0

        if w == 0 or h == 0:
            return 0.0, 0.0, float(w), float(h)

        if zf <= 1.0:
            if view_name == "axial":
                self.axial_center_x, self.axial_center_y = w / 2.0, h / 2.0
            elif view_name == "coronal":
                self.cor_center_x, self.cor_center_z = w / 2.0, h / 2.0
            elif view_name == "sagittal":
                self.sag_center_y, self.sag_center_z = w / 2.0, h / 2.0
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
            self.axial_center_x, self.axial_center_y = new_cx, new_cy
        elif view_name == "coronal":
            self.cor_center_x, self.cor_center_z = new_cx, new_cy
        elif view_name == "sagittal":
            self.sag_center_y, self.sag_center_z = new_cx, new_cy

        return left, top, width_view, height_view

    # -----------------------------------------------------
    # Drawing
    # -----------------------------------------------------
    def _draw_ui(self, view_name, slice_idx):
        self.main_ui.clear()
        if self.volume3d is None:
            return

        self.main_ui.stroke_style = 'yellow'
        self.main_ui.line_width = 2

        for box in self.box_prompts:
            if box['view'] == view_name and box['slice'] == slice_idx:
                x0,y0,x1,y1 = box['x0'], box['y0'], box['x1'], box['y1']
                x=min(x0,x1); y=min(y0,y1)
                w=abs(x1-x0); h=abs(y1-y0)
                self.main_ui.stroke_rect(x,y,w,h)

        self.main_ui.fill_style = 'cyan'
        for p in self.point_prompts:
            if p['view'] == view_name and p['slice'] == slice_idx:
                self.main_ui.begin_path()
                self.main_ui.arc(p['x'], p['y'], 4, 0, 2*math.pi)
                self.main_ui.fill()

    def _draw_mask_overlay_zoom_le1(self, mask_slice, img_size, offset):
        overlay = np.zeros((VIEW_SIZE, VIEW_SIZE, 4), dtype=np.uint8)
        if mask_slice is None or self.mask3d is None:
            return overlay

        mask_int = mask_slice.astype(np.int32)
        labels = np.unique(mask_int)
        labels = labels[labels > 0]
        if labels.size == 0:
            return overlay

        for lab in labels:
            lab = int(lab)
            if not self.label_visibility.get(lab, True):
                continue
            binary = (mask_int == lab).astype(np.uint8) * 255
            mask_img = Image.fromarray(binary).resize((img_size, img_size), Image.NEAREST)
            mask_np = np.array(mask_img) > 0
            r, g, b = self.hex_to_rgb(self.get_label_color(lab))
            sub = overlay[offset:offset+img_size, offset:offset+img_size]
            sub[mask_np, 0] = r
            sub[mask_np, 1] = g
            sub[mask_np, 2] = b
            sub[mask_np, 3] = 120
            overlay[offset:offset+img_size, offset:offset+img_size] = sub
        return overlay

    def _draw_mask_overlay_zoom_gt1(self, mask_slice, view_name, y0, y1, x0, x1):
        overlay = np.zeros((VIEW_SIZE, VIEW_SIZE, 4), dtype=np.uint8)
        if mask_slice is None or self.mask3d is None:
            return overlay

        sub = mask_slice[y0:y1, x0:x1].astype(np.int32)
        labels = np.unique(sub)
        labels = labels[labels > 0]
        if labels.size == 0:
            return overlay

        for lab in labels:
            lab = int(lab)
            if not self.label_visibility.get(lab, True):
                continue
            binary = (sub == lab).astype(np.uint8) * 255
            mask_img = Image.fromarray(binary).resize((VIEW_SIZE, VIEW_SIZE), Image.NEAREST)
            mask_np = np.array(mask_img) > 0
            r, g, b = self.hex_to_rgb(self.get_label_color(lab))
            overlay[mask_np, 0] = r
            overlay[mask_np, 1] = g
            overlay[mask_np, 2] = b
            overlay[mask_np, 3] = 120
        return overlay

    def _draw_slice_zoomed(self, slice2d, mask_slice, view_name, slice_idx):
        self.main_bg.clear()
        self.main_ol.clear()
        self.main_ui.clear()
        if self.volume3d is None:
            return

        zf = self.get_zoom_factor()

        if zf <= 1.0:
            h, w = slice2d.shape
            img_size = int(VIEW_SIZE * zf)
            if img_size <= 0:
                return

            rgba = np.zeros((VIEW_SIZE, VIEW_SIZE, 4), dtype=np.uint8)
            rgba[..., 3] = 255

            img8 = (slice2d * 255).astype(np.uint8)
            img = Image.fromarray(img8).resize((img_size, img_size), Image.BILINEAR)
            rgb = np.array(img.convert("RGB"), dtype=np.uint8)

            offset = (VIEW_SIZE - img_size) // 2
            rgba[offset:offset+img_size, offset:offset+img_size, :3] = rgb
            self.main_bg.put_image_data(rgba, 0, 0)

            if self.mask3d is not None and mask_slice is not None:
                overlay = self._draw_mask_overlay_zoom_le1(mask_slice, img_size, offset)
                self.main_ol.put_image_data(overlay, 0, 0)

        else:
            if view_name == "axial":
                h, w = self.dim_y, self.dim_x
            elif view_name == "coronal":
                h, w = self.dim_z, self.dim_x
            else:
                h, w = self.dim_z, self.dim_y

            left, top, width_view, height_view = self.get_view_window(view_name)
            x0 = int(max(0, min(w-1, math.floor(left))))
            x1 = int(max(x0+1, min(w, math.ceil(left+width_view))))
            y0 = int(max(0, min(h-1, math.floor(top))))
            y1 = int(max(y0+1, min(h, math.ceil(top+height_view))))

            sub = slice2d[y0:y1, x0:x1]
            img8 = (sub * 255).astype(np.uint8)
            img = Image.fromarray(img8).resize((VIEW_SIZE, VIEW_SIZE), Image.BILINEAR)
            rgb = np.array(img.convert("RGB"), dtype=np.uint8)

            rgba = np.zeros((VIEW_SIZE, VIEW_SIZE, 4), dtype=np.uint8)
            rgba[..., :3] = rgb
            rgba[..., 3] = 255
            self.main_bg.put_image_data(rgba, 0, 0)

            if self.mask3d is not None and mask_slice is not None:
                overlay = self._draw_mask_overlay_zoom_gt1(mask_slice, view_name, y0, y1, x0, x1)
                self.main_ol.put_image_data(overlay, 0, 0)

        self._draw_ui(view_name, slice_idx)

    def draw_axial(self):
        if self.volume3d is None:
            self.main_bg.clear(); self.main_ol.clear(); self.main_ui.clear()
            return
        slice2d = self.volume3d[self.axial_index]
        mask_slice = self.mask3d[self.axial_index] if self.mask3d is not None else None
        self._draw_slice_zoomed(slice2d, mask_slice, "axial", self.axial_index)

    def draw_coronal(self):
        if self.volume3d is None:
            self.main_bg.clear(); self.main_ol.clear(); self.main_ui.clear()
            return
        slice2d = self.volume3d[:, self.coronal_index, :]
        mask_slice = self.mask3d[:, self.coronal_index, :] if self.mask3d is not None else None
        self._draw_slice_zoomed(slice2d, mask_slice, "coronal", self.coronal_index)

    def draw_sagittal(self):
        if self.volume3d is None:
            self.main_bg.clear(); self.main_ol.clear(); self.main_ui.clear()
            return
        slice2d = self.volume3d[:, :, self.sagittal_index]
        mask_slice = self.mask3d[:, :, self.sagittal_index] if self.mask3d is not None else None
        self._draw_slice_zoomed(slice2d, mask_slice, "sagittal", self.sagittal_index)

    def draw_mip(self):
        self.main_bg.clear(); self.main_ol.clear(); self.main_ui.clear()
        if self.volume3d is None:
            return

        mip = self.volume3d.max(axis=0)
        img8 = (mip * 255).astype(np.uint8)
        img = Image.fromarray(img8)
        img = img.rotate(self.mip_angle, resample=Image.BILINEAR, expand=False)
        img = img.resize((VIEW_SIZE, VIEW_SIZE), Image.BILINEAR)
        rgb = np.array(img.convert("RGB"), dtype=np.uint8)

        rgba = np.zeros((VIEW_SIZE, VIEW_SIZE, 4), dtype=np.uint8)
        rgba[..., :3] = rgb
        rgba[..., 3] = 255
        self.main_bg.put_image_data(rgba, 0, 0)

        if self.mask3d is not None:
            mip_labels = self.mask3d.max(axis=0).astype(np.int32)
            labels = np.unique(mip_labels)
            labels = labels[labels > 0]
            if labels.size == 0:
                self.main_ol.clear()
                return

            mip_img = Image.fromarray(mip_labels.astype(np.uint8))
            mip_img = mip_img.rotate(self.mip_angle, resample=Image.NEAREST, expand=False)
            mip_img = mip_img.resize((VIEW_SIZE, VIEW_SIZE), Image.NEAREST)
            mip_resized = np.array(mip_img, dtype=np.int32)

            overlay = np.zeros((VIEW_SIZE, VIEW_SIZE, 4), dtype=np.uint8)
            for lab in labels:
                lab = int(lab)
                if not self.label_visibility.get(lab, True):
                    continue
                mask_np = (mip_resized == lab)
                r, g, b = self.hex_to_rgb(self.get_label_color(lab))
                overlay[mask_np, 0] = r
                overlay[mask_np, 1] = g
                overlay[mask_np, 2] = b
                overlay[mask_np, 3] = 120

            self.main_ol.put_image_data(overlay, 0, 0)

    def redraw(self):
        if self.current_view == "axial":
            self.draw_axial()
        elif self.current_view == "coronal":
            self.draw_coronal()
        elif self.current_view == "sagittal":
            self.draw_sagittal()
        else:
            self.draw_mip()

    # -----------------------------------------------------
    # Canvas→voxel mapping
    # -----------------------------------------------------
    def canvas_to_voxel_axial(self, cx, cy):
        if self.dim_x == 0 or self.dim_y == 0:
            return None, None
        zf = self.get_zoom_factor()
        if zf <= 1.0:
            img_size = int(VIEW_SIZE * zf)
            if img_size <= 0:
                return None, None
            offset = (VIEW_SIZE - img_size) / 2.0
            if not (offset <= cx < offset + img_size and offset <= cy < offset + img_size):
                return None, None
            u = (cx - offset) / img_size
            v = (cy - offset) / img_size
            vx = int(np.clip(round(u * (self.dim_x - 1)), 0, self.dim_x - 1))
            vy = int(np.clip(round(v * (self.dim_y - 1)), 0, self.dim_y - 1))
            return vx, vy
        else:
            left, top, width_view, height_view = self.get_view_window("axial")
            vx = left + (cx / VIEW_SIZE) * width_view
            vy = top + (cy / VIEW_SIZE) * height_view
            vx = int(np.clip(round(vx), 0, self.dim_x - 1))
            vy = int(np.clip(round(vy), 0, self.dim_y - 1))
            return vx, vy

    def canvas_to_voxel_coronal(self, cx, cy):
        if self.dim_x == 0 or self.dim_z == 0:
            return None, None
        zf = self.get_zoom_factor()
        if zf <= 1.0:
            img_size = int(VIEW_SIZE * zf)
            if img_size <= 0:
                return None, None
            offset = (VIEW_SIZE - img_size) / 2.0
            if not (offset <= cx < offset + img_size and offset <= cy < offset + img_size):
                return None, None
            u = (cx - offset) / img_size
            v = (cy - offset) / img_size
            vx = int(np.clip(round(u * (self.dim_x - 1)), 0, self.dim_x - 1))
            vz = int(np.clip(round(v * (self.dim_z - 1)), 0, self.dim_z - 1))
            return vz, vx
        else:
            left, top, width_view, height_view = self.get_view_window("coronal")
            vx = left + (cx / VIEW_SIZE) * width_view
            vz = top + (cy / VIEW_SIZE) * height_view
            vx = int(np.clip(round(vx), 0, self.dim_x - 1))
            vz = int(np.clip(round(vz), 0, self.dim_z - 1))
            return vz, vx

    def canvas_to_voxel_sagittal(self, cx, cy):
        if self.dim_y == 0 or self.dim_z == 0:
            return None, None
        zf = self.get_zoom_factor()
        if zf <= 1.0:
            img_size = int(VIEW_SIZE * zf)
            if img_size <= 0:
                return None, None
            offset = (VIEW_SIZE - img_size) / 2.0
            if not (offset <= cx < offset + img_size and offset <= cy < offset + img_size):
                return None, None
            u = (cx - offset) / img_size
            v = (cy - offset) / img_size
            vy = int(np.clip(round(u * (self.dim_y - 1)), 0, self.dim_y - 1))
            vz = int(np.clip(round(v * (self.dim_z - 1)), 0, self.dim_z - 1))
            return vz, vy
        else:
            left, top, width_view, height_view = self.get_view_window("sagittal")
            vy = left + (cx / VIEW_SIZE) * width_view
            vz = top + (cy / VIEW_SIZE) * height_view
            vy = int(np.clip(round(vy), 0, self.dim_y - 1))
            vz = int(np.clip(round(vz), 0, self.dim_z - 1))
            return vz, vy

    def canvas_to_image_xy(self, view_name, cx, cy):
        if view_name == "axial":
            vx, vy = self.canvas_to_voxel_axial(cx, cy)
            if vx is None: return None, None
            return vx, vy
        elif view_name == "coronal":
            vz, vx = self.canvas_to_voxel_coronal(cx, cy)
            if vz is None: return None, None
            return vx, vz
        elif view_name == "sagittal":
            vz, vy = self.canvas_to_voxel_sagittal(cx, cy)
            if vz is None: return None, None
            return vy, vz
        else:
            return None, None

    # -----------------------------------------------------
    # Painting (multi-label)
    # -----------------------------------------------------
    def paint_circle(self, cx, cy):
        if self.mask3d is None:
            return
        tool = self.tool_selector.value
        if tool not in ("Paint", "Erase"):
            return

        if self.current_view == "axial":
            vx, vy = self.canvas_to_voxel_axial(cx, cy)
            if vx is None: return
            z = self.axial_index
            scale = (self.dim_x/VIEW_SIZE + self.dim_y/VIEW_SIZE)/2
            r_vox = max(1, int(self.brush_size_slider.value * scale))
            y0=max(0,vy-r_vox); y1=min(self.dim_y, vy+r_vox+1)
            x0=max(0,vx-r_vox); x1=min(self.dim_x, vx+r_vox+1)
            yy,xx = np.ogrid[y0:y1, x0:x1]
            region = (yy-vy)**2 + (xx-vx)**2 <= r_vox*r_vox
            if tool == "Paint":
                self.mask3d[z,y0:y1,x0:x1][region]=self.current_label
            else:
                self.mask3d[z,y0:y1,x0:x1][region]=0

        elif self.current_view == "coronal":
            vz, vx = self.canvas_to_voxel_coronal(cx, cy)
            if vz is None: return
            y = self.coronal_index
            scale = (self.dim_x/VIEW_SIZE + self.dim_z/VIEW_SIZE)/2
            r_vox = max(1, int(self.brush_size_slider.value*scale))
            z0=max(0,vz-r_vox); z1=min(self.dim_z, vz+r_vox+1)
            x0=max(0,vx-r_vox); x1=min(self.dim_x, vx+r_vox+1)
            zz,xx = np.ogrid[z0:z1, x0:x1]
            region = (zz-vz)**2 + (xx-vx)**2 <= r_vox*r_vox
            if tool == "Paint":
                self.mask3d[z0:z1,y,x0:x1][region]=self.current_label
            else:
                self.mask3d[z0:z1,y,x0:x1][region]=0

        elif self.current_view == "sagittal":
            vz, vy = self.canvas_to_voxel_sagittal(cx, cy)
            if vz is None: return
            x = self.sagittal_index
            scale = (self.dim_y/VIEW_SIZE + self.dim_z/VIEW_SIZE)/2
            r_vox = max(1, int(self.brush_size_slider.value*scale))
            z0=max(0,vz-r_vox); z1=min(self.dim_z, vz+r_vox+1)
            y0=max(0,vy-r_vox); y1=min(self.dim_y, vy+r_vox+1)
            zz,yy=np.ogrid[z0:z1, y0:y1]
            region = (zz-vz)**2 + (yy-vy)**2 <= r_vox*r_vox
            if tool == "Paint":
                self.mask3d[z0:z1,y0:y1,x][region]=self.current_label
            else:
                self.mask3d[z0:z1,y0:y1,x][region]=0

        self.redraw()

    def paint_line(self, x0,y0,x1,y1):
        steps = int(max(abs(x1-x0),abs(y1-y0))/max(self.brush_size_slider.value/2,1))+1
        for t in np.linspace(0,1,steps):
            self.paint_circle(x0+t*(x1-x0), y0+t*(y1-y0))

    # -----------------------------------------------------
    # Hand / pan
    # -----------------------------------------------------
    def pan_view(self, view_name, dx_canvas, dy_canvas):
        if self.zoom_slider.value <= 100:
            return

        left, top, width_view, height_view = self.get_view_window(view_name)
        if width_view <= 0 or height_view <= 0:
            return

        dx_slice = -dx_canvas / VIEW_SIZE * width_view
        dy_slice = -dy_canvas / VIEW_SIZE * height_view

        if view_name == "axial":
            self.axial_center_x += dx_slice
            self.axial_center_y += dy_slice
        elif view_name == "coronal":
            self.cor_center_x += dx_slice
            self.cor_center_z += dy_slice
        elif view_name == "sagittal":
            self.sag_center_y += dx_slice
            self.sag_center_z += dy_slice

    # -----------------------------------------------------
    # Mouse events
    # -----------------------------------------------------
    def get_current_slice_index(self):
        if self.current_view == "axial":
            return self.axial_index
        if self.current_view == "coronal":
            return self.coronal_index
        if self.current_view == "sagittal":
            return self.sagittal_index
        return self.axial_index

    def on_mouse_down(self, x, y):
        if self.volume3d is None:
            return

        tool = self.tool_selector.value

        if self.current_view=="mip" and tool in ("Paint", "Erase", "Box", "Point"):
            return

        if tool in ("Paint", "Erase"):
            self.drawing=True
            self.last_canvas_x, self.last_canvas_y = x,y
            self.paint_circle(x,y)

        elif tool=="Box":
            slice_idx = self.get_current_slice_index()
            ix0, iy0 = self.canvas_to_image_xy(self.current_view, x, y)
            if ix0 is None:
                return
            self.box_prompts.append({
                'view': self.current_view,
                'slice': slice_idx,
                'x0':x, 'y0':y, 'x1':x, 'y1':y,
                'ix0':ix0, 'iy0':iy0, 'ix1':ix0, 'iy1':iy0
            })
            self.box_drawing=True
            self._draw_ui(self.current_view, slice_idx)

        elif tool=="Point":
            slice_idx = self.get_current_slice_index()
            ix, iy = self.canvas_to_image_xy(self.current_view, x, y)
            if ix is None:
                return
            self.point_prompts.append({
                'view':self.current_view,
                'slice':slice_idx,
                'x':x,'y':y,
                'ix':ix,'iy':iy
            })
            self._draw_ui(self.current_view, slice_idx)

        elif tool=="Hand":
            self.drawing = True
            self.last_canvas_x, self.last_canvas_y = x, y

    def on_mouse_move(self, x, y):
        if self.volume3d is None:
            return

        tool = self.tool_selector.value

        if tool in ("Paint", "Erase") and self.drawing and self.current_view != "mip":
            self.paint_line(self.last_canvas_x, self.last_canvas_y, x, y)
            self.last_canvas_x, self.last_canvas_y = x,y

        elif tool=="Box" and self.box_drawing and self.current_view != "mip":
            if self.box_prompts:
                self.box_prompts[-1]['x1']=x
                self.box_prompts[-1]['y1']=y
                ix1, iy1 = self.canvas_to_image_xy(self.current_view, x, y)
                if ix1 is not None:
                    self.box_prompts[-1]['ix1'] = ix1
                    self.box_prompts[-1]['iy1'] = iy1
                self._draw_ui(self.current_view, self.get_current_slice_index())

        elif tool=="Hand" and self.drawing:
            dx = x - self.last_canvas_x
            dy = y - self.last_canvas_y
            if self.current_view == "mip":
                self.mip_angle += dx * 0.3
                self.last_canvas_x, self.last_canvas_y = x, y
                self.redraw()
            else:
                self.pan_view(self.current_view, dx, dy)
                self.last_canvas_x, self.last_canvas_y = x, y
                self.redraw()

    def on_mouse_up(self, x, y):
        if self.volume3d is None:
            self.drawing=False; self.box_drawing=False
            return

        tool = self.tool_selector.value

        if tool in ("Paint", "Erase"):
            self.drawing=False

        elif tool=="Box" and self.box_drawing:
            if self.current_view != "mip" and self.box_prompts:
                self.box_prompts[-1]['x1']=x
                self.box_prompts[-1]['y1']=y
                ix1, iy1 = self.canvas_to_image_xy(self.current_view, x, y)
                if ix1 is not None:
                    self.box_prompts[-1]['ix1'] = ix1
                    self.box_prompts[-1]['iy1'] = iy1
                self._draw_ui(self.current_view, self.get_current_slice_index())
            self.box_drawing=False

        elif tool=="Hand":
            self.drawing=False

    # -----------------------------------------------------
    # Load MRI / model / mask
    # -----------------------------------------------------
    def load_from_path(self, _chooser):
        path = self.fc.selected
        if not path:
            self.log_status("Please choose an MRI file first.")
            return

        path = os.path.abspath(path)
        if not os.path.isfile(path):
            self.log_status(f"File not found: {os.path.basename(path)}")
            return
        fname = os.path.basename(path)

        try:
            img_local = sitk.ReadImage(path)
        except Exception as e:
            self.mri_load_label.value = self._status_html("MRI", False)
            self.log_status(f"SimpleITK read failed: {e}")
            return

        try:
            vol = self.prepare_volume(img_local)
        except Exception as e:
            self.mri_load_label.value = self._status_html("MRI", False)
            self.log_status(f"Normalization failed: {e}")
            return

        self.volume3d = vol
        self.image_sitk = img_local
        self.current_image_path = path
        self.mri_load_label.value = self._status_html("MRI", True, fname)

        self.dim_z, self.dim_y, self.dim_x = self.volume3d.shape
        self.init_mask()

        self.axial_index = self.dim_z // 2
        self.coronal_index = self.dim_y // 2
        self.sagittal_index = self.dim_x // 2

        self.axial_center_x = self.dim_x / 2.0
        self.axial_center_y = self.dim_y / 2.0
        self.cor_center_x = self.dim_x / 2.0
        self.cor_center_z = self.dim_z / 2.0
        self.sag_center_y = self.dim_y / 2.0
        self.sag_center_z = self.dim_z / 2.0

        self.mip_angle = 0.0
        self.zoom_slider.value = self.zoom_slider_value_default

        self.axial_slider.min=0; self.axial_slider.max=self.dim_z-1; self.axial_slider.value=self.axial_index; self.axial_slider.disabled=False
        self.coronal_slider.min=0; self.coronal_slider.max=self.dim_y-1; self.coronal_slider.value=self.coronal_index; self.coronal_slider.disabled=False
        self.sagittal_slider.min=0; self.sagittal_slider.max=self.dim_x-1; self.sagittal_slider.value=self.sagittal_index; self.sagittal_slider.disabled=False

        self.box_prompts=[]; self.point_prompts=[]
        self.drawing=False; self.box_drawing=False
        self.current_view="axial"

        self.last_sam_image_view = None
        self.last_sam_image_slice_idx = None
        self.sam_candidates = None
        self.candidate_thumbs_box.children = []

        stem, ext = self.split_name_and_ext(path)
        self.last_save_dir = os.path.dirname(path)
        self.save_filename_text.value = stem
        if ext in self.save_format_dropdown.options:
            self.save_format_dropdown.value = ext
        else:
            self.save_format_dropdown.value = '.nii.gz'
        self.save_dir_chooser.default_path = self.last_save_dir
        self.save_dir_chooser.reset()

        self.update_label_state_from_mask()
        self.redraw()
        self.log_status(f"Loaded MRI: {fname} (Z={self.dim_z},Y={self.dim_y},X={self.dim_x}). View: Axial.")

    def load_model(self, _chooser):
        path = self.fc_model.selected
        if not path:
            self.log_status("Please choose a SAMRI checkpoint file first.")
            return

        path = os.path.abspath(path)
        if not os.path.isfile(path):
            self.log_status(f"Model file not found: {os.path.basename(path)}")
            return
        fname = os.path.basename(path)

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
            self.model_load_label.value = self._status_html("SAMRI", False)
            self.log_status(f"Failed to load SAMRI model: {e}")
            return

        self.current_model_path = path
        self.sam_model = model
        self.sam_predictor = predictor
        self.sam_device = device

        self.last_sam_image_view = None
        self.last_sam_image_slice_idx = None
        self.sam_candidates = None
        self.candidate_thumbs_box.children = []
        self.model_load_label.value = self._status_html("SAMRI", True, fname)

        self.log_status(f"SAMRI model loaded: {fname} on device '{device}'.")

    def load_mask(self, _chooser):
        if self.volume3d is None:
            self.mask_load_label.value = self._status_html("Mask", False)
            self.log_status("Please load an MRI before loading a mask.")
            return

        path = self.fc_mask.selected
        if not path:
            self.mask_load_label.value = self._status_html("Mask", False)
            self.log_status("Please choose a mask file first.")
            return

        path = os.path.abspath(path)
        if not os.path.isfile(path):
            self.mask_load_label.value = self._status_html("Mask", False)
            self.log_status(f"Mask file not found: {os.path.basename(path)}")
            return

        fname = os.path.basename(path)

        try:
            mask_img = sitk.ReadImage(path)
            mask_arr = sitk.GetArrayFromImage(mask_img)
        except Exception as e:
            self.mask_load_label.value = self._status_html("Mask", False)
            self.log_status(f"Failed to read mask file: {e}")
            return

        if mask_arr.ndim == 2:
            self.mask_load_label.value = self._status_html("Mask", False)
            self.log_status(
                f"Loaded mask is 2D with shape {mask_arr.shape}. Expected 3D mask "
                f"with shape {self.volume3d.shape} (Z, Y, X)."
            )
            return
        elif mask_arr.ndim == 3:
            pass
        elif mask_arr.ndim == 4:
            mask_arr = mask_arr[0]
        else:
            self.mask_load_label.value = self._status_html("Mask", False)
            self.log_status(f"Unsupported mask dimensions: {mask_arr.shape}")
            return

        if mask_arr.shape != self.volume3d.shape:
            self.mask_load_label.value = self._status_html("Mask", False)
            self.log_status(
                f"Mask shape {mask_arr.shape} does not match MRI shape {self.volume3d.shape} (Z, Y, X)."
            )
            return

        self.mask3d = mask_arr.astype(np.uint8)

        stem, ext = self.split_name_and_ext(path)
        self.last_save_dir = os.path.dirname(path)
        self.save_dir_chooser.default_path = self.last_save_dir
        self.save_dir_chooser.reset()
        if stem:
            self.save_filename_text.value = stem
        if ext in self.save_format_dropdown.options:
            self.save_format_dropdown.value = ext

        self.mask_load_label.value = self._status_html("Mask", True, fname)
        self.update_label_state_from_mask()
        self.redraw()
        self.log_status(f"Mask loaded from {fname} and applied to volume.")

    # -----------------------------------------------------
    # Clear / save / prompts
    # -----------------------------------------------------
    def clear_prompts(self, _):
        if self.volume3d is None:
            return
        self.box_prompts = []
        self.point_prompts = []
        self.box_drawing = False
        self.sam_candidates = None
        self.candidate_thumbs_box.children = []
        self._draw_ui(self.current_view, self.get_current_slice_index())
        self.log_status("Cleared prompts and candidate masks.")

    def clear_mask(self, _):
        if self.volume3d is None or self.mask3d is None:
            return

        labels_to_clear = [lab for lab, vis in self.label_visibility.items() if vis]
        if not labels_to_clear:
            self.log_status("No labels selected (visible) to clear.")
            return

        for lab in labels_to_clear:
            if lab == 0:
                continue
            self.mask3d[self.mask3d == lab] = 0

        self.update_label_state_from_mask()
        self.redraw()
        self.log_status(f"Cleared labels: {labels_to_clear}")

    def save_mask(self, _):
        if self.volume3d is None or self.mask3d is None:
            self.log_status("Nothing to save (no volume or mask).")
            return

        dir_path = self.save_dir_chooser.selected or self.last_save_dir or os.getcwd()
        dir_path = os.path.abspath(dir_path)
        if not os.path.isdir(dir_path):
            self.log_status(f"Invalid save directory: {dir_path}")
            return

        stem = self.save_filename_text.value.strip()
        if not stem:
            stem = "mask3d"
        ext = self.save_format_dropdown.value
        filename = stem + ext
        save_path = os.path.join(dir_path, filename)

        mask_arr = self.mask3d.astype(np.uint8)

        if ext == '.png':
            if self.current_view == "axial":
                mask2d = mask_arr[self.axial_index]
            elif self.current_view == "coronal":
                mask2d = mask_arr[:, self.coronal_index, :]
            elif self.current_view == "sagittal":
                mask2d = mask_arr[:, :, self.sagittal_index]
            else:
                mask2d = mask_arr[self.axial_index]

            img2d = mask2d.astype(np.uint8)
            try:
                Image.fromarray(img2d).save(save_path)
            except Exception as e:
                self.log_status(f"Failed to save PNG mask: {e}")
                return
        else:
            mask_img = sitk.GetImageFromArray(mask_arr)
            if self.image_sitk is not None:
                mask_img.CopyInformation(self.image_sitk)
            try:
                sitk.WriteImage(mask_img, save_path)
            except Exception as e:
                self.log_status(f"Failed to save mask: {e}")
                return

        self.last_save_dir = dir_path
        self.save_dir_chooser.default_path = self.last_save_dir
        self.save_dir_chooser.reset()
        self.log_status(f"Mask saved to {save_path}")

    # -----------------------------------------------------
    # SAMRI inference
    # -----------------------------------------------------
    def get_current_slice_2d(self):
        if self.current_view == "axial":
            return self.volume3d[self.axial_index]
        elif self.current_view == "coronal":
            return self.volume3d[:, self.coronal_index, :]
        elif self.current_view == "sagittal":
            return self.volume3d[:, :, self.sagittal_index]
        else:
            return self.volume3d[self.axial_index]

    @staticmethod
    def make_sam_input_from_slice(slice2d):
        img8 = (slice2d * 255.0).clip(0, 255).astype(np.uint8)
        img_rgb = np.stack([img8, img8, img8], axis=-1)
        return img_rgb

    def write_label_mask2d_to_volume(self, view_name, slice_idx, binary_mask2d, label_id):
        if label_id < 0 or self.mask3d is None:
            return False

        if view_name == "axial":
            if binary_mask2d.shape != (self.dim_y, self.dim_x):
                return False
            slice_view = self.mask3d[slice_idx, :, :]
            slice_view[slice_view == label_id] = 0
            slice_view[binary_mask2d > 0] = label_id
            self.mask3d[slice_idx, :, :] = slice_view
            return True

        elif view_name == "coronal":
            if binary_mask2d.shape != (self.dim_z, self.dim_x):
                return False
            slice_view = self.mask3d[:, slice_idx, :]
            slice_view[slice_view == label_id] = 0
            slice_view[binary_mask2d > 0] = label_id
            self.mask3d[:, slice_idx, :] = slice_view
            return True

        elif view_name == "sagittal":
            if binary_mask2d.shape != (self.dim_z, self.dim_y):
                return False
            slice_view = self.mask3d[:, :, slice_idx]
            slice_view[slice_view == label_id] = 0
            slice_view[binary_mask2d > 0] = label_id
            self.mask3d[:, :, slice_idx] = slice_view
            return True

        return False

    def apply_candidate_mask(self, idx):
        if self.sam_candidates is None:
            self.log_status("No candidate masks available.")
            return

        view   = self.sam_candidates['view']
        sl     = self.sam_candidates['slice_idx']
        masks  = self.sam_candidates['masks']
        scores = self.sam_candidates['scores']
        label_id = int(self.sam_candidates.get('label_id', self.current_label))

        if idx < 0 or idx >= masks.shape[0]:
            self.log_status("Invalid candidate index.")
            return

        if self.current_view != view or self.get_current_slice_index() != sl:
            self.log_status("Candidate masks belong to a different slice/view. Regenerate on this slice to update.")
            return

        binary2d = (masks[idx] > 0).astype(np.uint8)
        ok = self.write_label_mask2d_to_volume(view, sl, binary2d, label_id)
        if not ok:
            self.log_status(f"Mask shape mismatch for view={view}. Got {binary2d.shape}.")
            return

        self.update_label_state_from_mask()
        self.redraw()
        self.log_status(
            f"Applied candidate mask {idx} (score={scores[idx]:.3f}) "
            f"as label {label_id} on {view} slice {sl}."
        )

    def make_mask_thumbnail(self, slice2d, mask2d, score, idx, color_hex, thumb_size=64):
        img8 = (slice2d * 255.0).clip(0, 255).astype(np.uint8)
        base = Image.fromarray(img8).convert("RGB")
        mask_arr = (mask2d > 0).astype(np.uint8)
        mask_img = Image.fromarray(mask_arr * 255).convert("L")

        base = base.resize((thumb_size, thumb_size), Image.BILINEAR)
        mask_img = mask_img.resize((thumb_size, thumb_size), Image.NEAREST)

        base_np = np.array(base)
        mask_np = np.array(mask_img) > 0

        r, g, b = self.hex_to_rgb(color_hex)
        overlay = base_np.copy()
        overlay[mask_np, 0] = r
        overlay[mask_np, 1] = g
        overlay[mask_np, 2] = b

        thumb = Image.fromarray(overlay.astype(np.uint8))
        bio = BytesIO()
        thumb.save(bio, format='PNG')
        png_bytes = bio.getvalue()

        img_widget = WImage(
            value=png_bytes,
            format='png',
            width=thumb_size,
            height=thumb_size
        )

        btn = Button(
            description=f"{idx}: {score:.3f}",
            layout={'width': f'{thumb_size + 20}px'}
        )

        def _on_click(_b, i=idx):
            self.apply_candidate_mask(i)

        btn.on_click(_on_click)
        return VBox([img_widget, btn])

    def update_candidate_thumbnails(self, slice2d, masks, scores, color_hex):
        if masks is None or masks.ndim != 3:
            self.candidate_thumbs_box.children = []
            return
        K = masks.shape[0]
        thumbs = [
            self.make_mask_thumbnail(slice2d, masks[i], float(scores[i]), i, color_hex)
            for i in range(K)
        ]
        self.candidate_thumbs_box.children = thumbs

    def on_generate_mask(self, _):
        if self.volume3d is None:
            self.log_status("No MRI loaded. Please load an image first.")
            return
        if self.sam_predictor is None:
            self.log_status("No SAMRI model loaded. Please load a checkpoint first.")
            return
        if self.current_view == "mip":
            self.log_status("Generate Mask works on Axial/Coronal/Sagittal slices, not MIP.")
            return

        slice_idx = self.get_current_slice_index()
        slice2d = self.get_current_slice_2d()
        img_rgb = self.make_sam_input_from_slice(slice2d)

        if not (self.last_sam_image_view == self.current_view and
                self.last_sam_image_slice_idx == slice_idx):
            try:
                self.sam_predictor.set_image(img_rgb)
            except Exception as e:
                self.log_status(f"SAMRI set_image failed: {e}")
                return
            self.last_sam_image_view = self.current_view
            self.last_sam_image_slice_idx = slice_idx

        H, W = img_rgb.shape[:2]

        boxes_here = [b for b in self.box_prompts
                      if b['view'] == self.current_view and b['slice'] == slice_idx]
        points_here = [p for p in self.point_prompts
                       if p['view'] == self.current_view and p['slice'] == slice_idx]

        box_array = None
        if boxes_here:
            b = boxes_here[-1]
            x0 = float(b['ix0']); y0 = float(b['iy0'])
            x1 = float(b['ix1']); y1 = float(b['iy1'])
            x_min = max(0.0, min(x0, x1))
            y_min = max(0.0, min(y0, y1))
            x_max = min(float(W - 1), max(x0, x1))
            y_max = min(float(H - 1), max(y0, y1))
            box_array = np.array([[x_min, y_min, x_max, y_max]], dtype=np.float32)

        point_coords = None
        point_labels = None
        if points_here:
            coords = []
            labels = []
            for p in points_here:
                ix = float(p['ix']); iy = float(p['iy'])
                if 0 <= ix < W and 0 <= iy < H:
                    coords.append([ix, iy]); labels.append(1)
            if coords:
                point_coords = np.array(coords, dtype=np.float32)
                point_labels = np.array(labels, dtype=np.int32)

        if box_array is None and point_coords is None:
            self.log_status("No prompts on this slice. Please draw a box or point first.")
            return

        try:
            masks, scores, logits = self.sam_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box_array,
                multimask_output=True
            )
        except Exception as e:
            self.log_status(f"SAMRI predict failed: {e}")
            return

        if masks.ndim != 3:
            self.log_status(f"Unexpected masks shape: {masks.shape}")
            return

        label_at_gen = self.current_label
        color_at_gen = self.get_label_color(label_at_gen)

        self.sam_candidates = {
            'view': self.current_view,
            'slice_idx': slice_idx,
            'masks': masks,
            'scores': scores,
            'label_id': label_at_gen,
            'color_hex': color_at_gen
        }

        self.update_candidate_thumbnails(slice2d, masks, scores, color_at_gen)

        K = masks.shape[0]
        best_idx = int(np.argmax(scores))

        self.apply_candidate_mask(best_idx)
        self.log_status(
            f"Generated {K} candidate masks on {self.current_view} slice {slice_idx} "
            f"for label {label_at_gen}. Best idx={best_idx} (score={scores[best_idx]:.3f}). "
            f"Click thumbnails to switch masks."
        )

    # -----------------------------------------------------
    # Slider / label / zoom callbacks
    # -----------------------------------------------------
    def on_axial_change(self, change):
        self.axial_index = change['new']
        if self.current_view=="axial":
            self.draw_axial()

    def on_coronal_change(self, change):
        self.coronal_index = change['new']
        if self.current_view=="coronal":
            self.draw_coronal()

    def on_sag_change(self, change):
        self.sagittal_index = change['new']
        if self.current_view=="sagittal":
            self.draw_sagittal()

    def axial_minus(self, _):
        if self.axial_slider.disabled:
            return
        self.axial_slider.value = max(self.axial_slider.min, self.axial_slider.value - 1)

    def axial_plus(self, _):
        if self.axial_slider.disabled:
            return
        self.axial_slider.value = min(self.axial_slider.max, self.axial_slider.value + 1)

    def coronal_minus(self, _):
        if self.coronal_slider.disabled:
            return
        self.coronal_slider.value = max(self.coronal_slider.min, self.coronal_slider.value - 1)

    def coronal_plus(self, _):
        if self.coronal_slider.disabled:
            return
        self.coronal_slider.value = min(self.coronal_slider.max, self.coronal_slider.value + 1)

    def sagittal_minus(self, _):
        if self.sagittal_slider.disabled:
            return
        self.sagittal_slider.value = max(self.sagittal_slider.min, self.sagittal_slider.value - 1)

    def sagittal_plus(self, _):
        if self.sagittal_slider.disabled:
            return
        self.sagittal_slider.value = min(self.sagittal_slider.max, self.sagittal_slider.value + 1)

    def on_label_color_change(self, change):
        self.label_colors[self.current_label] = change['new']
        self.update_label_legend()
        if self.volume3d is not None:
            self.redraw()

    def on_current_label_input_change(self, change):
        if change['name'] != 'value':
            return
        new_label = int(change['new'])
        if new_label < 0:
            new_label = 0
            self.current_label_input.value = 0
        self.current_label = new_label
        self.get_label_color(self.current_label)
        self.label_color_picker.value = self.get_label_color(self.current_label)
        self.update_label_legend()
        if self.volume3d is not None:
            self.redraw()

    def on_add_label_clicked(self, _):
        labels = set(self.label_colors.keys())
        if self.mask3d is not None:
            labels.update(np.unique(self.mask3d).tolist())
        labels.discard(0)
        if not labels:
            new_label = 1
        else:
            new_label = max(labels) + 1
        self.current_label = int(new_label)
        self.get_label_color(self.current_label)
        self.current_label_input.value = self.current_label
        self.label_color_picker.value = self.get_label_color(self.current_label)
        self.update_label_legend()
        if self.volume3d is not None:
            self.redraw()
        self.log_status(f"Added new label: {self.current_label}")

    def on_delete_label_clicked(self, _):
        label_to_delete = self.current_label
        if label_to_delete == 0:
            self.log_status("Cannot delete background label 0.")
            return

        if self.mask3d is not None:
            self.mask3d[self.mask3d == label_to_delete] = 0

        if label_to_delete in self.label_colors:
            del self.label_colors[label_to_delete]
        if label_to_delete in self.label_names:
            del self.label_names[label_to_delete]
        if label_to_delete in self.label_visibility:
            del self.label_visibility[label_to_delete]
        if label_to_delete in self.label_checkboxes:
            del self.label_checkboxes[label_to_delete]

        labels = set(self.label_colors.keys())
        if self.mask3d is not None:
            labels.update(np.unique(self.mask3d).tolist())
        labels.discard(0)
        if not labels:
            labels = {1}
            self.get_label_color(1)
        self.current_label = sorted(labels)[0]
        self.current_label_input.value = self.current_label
        self.label_color_picker.value = self.get_label_color(self.current_label)
        self.update_label_legend()
        if self.volume3d is not None:
            self.redraw()
        self.log_status(f"Deleted label {label_to_delete} and cleared it from mask.")

    def on_all_labels_change(self, change):
        if change['name'] != 'value':
            return
        if self.legend_batch_update:
            return
        value = bool(change['new'])
        self.legend_batch_update = True
        for lab, cb in self.label_checkboxes.items():
            self.label_visibility[lab] = value
            cb.value = value
        self.legend_batch_update = False
        if self.volume3d is not None:
            self.redraw()

    def on_zoom_change(self, _):
        if self.volume3d is not None:
            self.redraw()

    def on_canvas_size_change(self, change):
        new_size = int(change['new'])
        if new_size <= 0:
            return
        self.main_canv.layout.width = f"{new_size}px"
        self.main_canv.layout.height = f"{new_size}px"
        self.log_status(f"Canvas display size set to {new_size} x {new_size}px")

    def zoom_out(self, _):
        self.zoom_slider.value = max(self.zoom_slider.min, self.zoom_slider.value - 10)

    def zoom_in(self, _):
        self.zoom_slider.value = min(self.zoom_slider.max, self.zoom_slider.value + 10)

    def canvas_size_minus(self, _):
        self.canvas_size_slider.value = max(
            self.canvas_size_slider.min,
            self.canvas_size_slider.value - self.canvas_size_slider.step
        )

    def canvas_size_plus(self, _):
        self.canvas_size_slider.value = min(
            self.canvas_size_slider.max,
            self.canvas_size_slider.value + self.canvas_size_slider.step
        )



