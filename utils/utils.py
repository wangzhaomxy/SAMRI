# -*- coding: utf-8 -*-

"""
Permanent variables, universal functions, ect
"""
from glob import glob
import os
from torch.nn import functional as F

root_path = "/scratch/project/samri/"
ch_root = "/scratch/user/s4670484/Model_dir/"
EMBEDDING_PATH = root_path + "Embedding/" # The main folder of datasets
TEST_PATH = root_path + "Datasets/SAMRI_train_test/"
MODEL_SAVE_PATH = root_path + "Model_save/"
DEVICE = "cuda"
BATCH_SIZE = 384
NUM_EPOCHS = 20
JITTER = 3

TRAIN_IMAGE_PATH = [ds + "/" for ds in sorted(glob(EMBEDDING_PATH + "*"))]
TEST_IMAGE_PATH = [ds + "/testing/" for ds in sorted(glob(TEST_PATH + "*"))]
TEST_IMAGE_PATH_DA = [ds + "/training/" for ds in sorted(glob(TEST_PATH + "*"))]

IMAGE_KEYS = "*_img_*"  # The image file names containing letters between *
MASK_KEYS = "*_seg_*"   # The mask file names containing letters between *

ENCODER_TYPE = {"vit_b":"vit_b",
                  "vit_h":"vit_h",
                  "med_sam":"vit_b",
                  "samri":"vit_b"
                  }

SAM_CHECKPOINT = {"vit_b": ch_root + "sam_vit_b_01ec64.pth",
                  "vit_h": ch_root + "sam_vit_h_4b8939.pth",
                  "med_sam": ch_root + "medsam_vit_b.pth",
                  "samri": ch_root + "samri_vitb.pth"
                  }


def get_checkpoint(path):
    cp_list = sorted(glob(path + "*pth"))
    cp_names = [(os.path.basename(cp)[:-4]) for cp in cp_list]
    start_epoch = max([int(cp.split('_')[-1]) for cp in cp_names if cp != ""])
    cp_name = glob(path + f"*_{str(start_epoch)}.pth*")[0]
    return cp_name, start_epoch

def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) :
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

def preprocess_mask(mask, target_size=256):
    """
    Preprocess masks from original size to target size.

    Args:
        mask (tensor): the mask with shape of BxCxHxW
        target_size (int, optional): The target size. Defaults to 256.
        
    Returns:
        (tensor): the mask with the target shape.
        
    """
    resize_long = get_preprocess_shape(mask.shape[-2], mask.shape[-1], target_size)
    resized_mask = F.interpolate(mask, 
                           resize_long, 
                           mode="nearest")
    # Pad
    h, w = resized_mask.shape[-2:]
    padh = target_size - h
    padw = target_size - w
    x = F.pad(resized_mask, (0, padw, 0, padh))
    return x