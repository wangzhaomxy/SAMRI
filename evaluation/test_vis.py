# -*- coding: utf-8 -*-
"""
Single-GPU testing script for SAMRI mask decoder.
Loads checkpoints from a specified path or directory.

Examples:
  - Test a single checkpoint:
      python evaluation.test_vis.py \
        --test-image-path /path/to/test/images/ \
        --ckpt-path /path/to/checkpoint.pth \
        --save-path /path/to/save/results/ \
        --device cuda \
        --model-type samri
        
  - Test all checkpoints in a directory:
      python evaluation.test_vis.py \
        --test-image-path /path/to/test/images/ \
        --ckpt-path /path/to/checkpoint_directory/
        --save-path /path/to/save/results/ \
        --device cuda \
        --model-type samri

  - Test only some datasets of a root (e.g. rerun SAM ViT-B on MSK_shoulder):
      python -m evaluation.test_vis \
        --test-image-path /path/to/Zeroshot/ \
        --datasets MSK_shoulder CHAOS_T1 \
        --ckpt-path /path/to/sam_vit_b_01ec64.pth \
        --save-path /path/to/save/results/ \
        --model-type vit_b
"""

from segment_anything import sam_model_registry
from utils.visual import *
from utils.utils import *
from pathlib import Path
import re, os
import time
import argparse

cfg = SAMRIConfig()
# setup global parameters and converted to CLI-driven.
_parser = argparse.ArgumentParser(add_help=True)
_parser.add_argument("--test-image-path", "--test_image_path",
                     dest="test_image_path",
                     type=str, 
                     default=cfg.IMAGE_PATH,
                     help="The root path of the images.")
_parser.add_argument("--ckpt-path", "--ckpt_path",
                     dest="ckpt_path",
                     type=str, 
                     default=cfg.MODEL_SAVE_PATH,
                     help="The root path or path of the test checkpoint.")
_parser.add_argument("--save-path", "--save_path",
                     dest="save_path",
                     type=str, 
                     default="/scratch/user/s4670484/Eval_results/",
                     help="The root path to save evaluation results.")
_parser.add_argument("--datasets", nargs="*", default=None,
                     help="Only these dataset folder names under --test-image-path (default: all).")
_parser.add_argument("--device",
                     dest="device",
                     type=str,
                     default=cfg.DEVICE,
                     help="Device to run the model on.")
_parser.add_argument("--model-type", "--model_type",
                     dest="model_type",
                     default="samri",
                     choices=list(cfg.ENCODER_TYPE.keys()),
                     help="Model key used to derive encoder_type from ENCODER_TYPE.")
_args = _parser.parse_args()
# cfg.IMAGE_PATH is a read-only property, so the test root is kept here instead of assigned to cfg
test_image_root = os.path.join(_args.test_image_path, "")  # glob below needs the trailing "/"
ckpt_path = _args.ckpt_path
save_path = _args.save_path
os.makedirs(save_path, exist_ok=True)

def test_image_path(path):
    """substract images from the main dataset folder

    Args:
        path (str): The root path of the datasets.

    Returns:
        list: A list of paths to the testing subdirectories.
    """
    return [
        ds + "/testing/" for ds in sorted(glob(path + "*"))
        if os.path.isdir(ds)
    ]
    
def load_ckpt_list(ckpt_dir):
    """
    Given a checkpoint path:
      - If ckpt_dir points to a single .pth file → return [absolute path].
      - If ckpt_dir is a directory → return sorted list of all .pth files.
    """
    ckpt_path = Path(ckpt_dir)

    # Case 1: single checkpoint file
    if ckpt_path.is_file() and ckpt_path.suffix == ".pth":
        return [str(ckpt_path.resolve())]

    # Case 2: directory containing checkpoints
    elif ckpt_path.is_dir():
        ckpt_paths = list(ckpt_path.glob("samri_vitb_box_*.pth"))

        # Extract numeric epoch from filenames
        def get_epoch_number(path):
            match = re.search(r"_(\d+)\.pth$", path.name)
            return int(match.group(1)) if match else -1

        # Sort numerically by epoch number
        ckpt_list = [str(p.resolve()) for p in sorted(ckpt_paths, key=get_epoch_number)]
        return ckpt_list

    else:
        raise FileNotFoundError(f"Checkpoint path not found: {ckpt_dir}")

# Load test checkpoint paths
ckpt_list = load_ckpt_list(ckpt_path)

print(f"Found {len(ckpt_list)} checkpoint(s):")
for ckpt in ckpt_list:
    print(ckpt)

for ckpt in ckpt_list:
    start = time.time()
    model_type = _args.model_type # Choose one from vit_b, vit_h, samri, and med_sam
    encoder_tpye = cfg.ENCODER_TYPE[model_type] 
    checkpoint = ckpt
    device = _args.device
    model_name = ckpt.split("/")[-1]
    print("Testing Check-point: " + ckpt)

    # regist the MRI-SAM model and predictor.
    sam_model = sam_model_registry[encoder_tpye](checkpoint)
    sam_model = sam_model.to(device)
    sam_model.eval()
    save_path_all = save_path + model_name[:-4]

    file_paths = test_image_path(test_image_root)
    if _args.datasets:
        file_paths = [p for p in file_paths if p.split("/")[-3] in _args.datasets]
        missing = set(_args.datasets) - {p.split("/")[-3] for p in file_paths}
        if missing:
            raise FileNotFoundError(f"Datasets not found under {test_image_root}: {sorted(missing)}")
    print("Testing datasets:", [p.split("/")[-3] for p in file_paths])
    save_test_record(file_paths=file_paths,
                     sam_model=sam_model, 
                     save_path=save_path_all)
    
    end = time.time()
    print(f"Elapsed time: {end - start:.2f} seconds")
    print("Done!")