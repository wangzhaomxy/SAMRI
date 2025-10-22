# -*- coding: utf-8 -*-

"""
Permanent variables, universal functions, ect
"""
# config_samri.py
import os
from torch.nn import functional as F
import numpy as np
import torch
import random
from glob import glob

class SAMRIConfig:
    def __init__(
        self,
        root_path="/scratch/project/samri/",
        ch_root="/scratch/user/s4670484/Model_dir/",
        device="cuda",
        batch_size=1024,
        num_epochs=200,
        jitter=10,
        image_keys="*_img_*",
        mask_keys="*_seg_*",
    ):
        # ---- user-tunable ----
        self.root_path = root_path
        self.ch_root = ch_root
        self.DEVICE = device
        self.BATCH_SIZE = batch_size
        self.NUM_EPOCHS = num_epochs
        self.JITTER = jitter
        self.IMAGE_KEYS = image_keys
        self.MASK_KEYS = mask_keys

    # =====================================================================
    #  Dynamically computed properties (auto-update when root_path changes)
    # =====================================================================

    @property
    def IMAGE_PATH(self):
        return self.root_path + "Datasets/SAMRI_train_test/"

    @property
    def MODEL_SAVE_PATH(self):
        return self.root_path + "Model_save/"

    @property
    def TRAIN_EMBEDDING_PATH(self):
        return [
            ds + "/" for ds in sorted(glob(self.root_path + "Datasets/Embedding_train/*"))
            if os.path.isdir(ds)
        ]

    @property
    def VAL_EMBEDDING_PATH(self):
        return [
            ds + "/" for ds in sorted(glob(self.root_path + "Datasets/Embedding_val/*"))
            if os.path.isdir(ds)
        ]

    @property
    def TRAIN_IMAGE_PATH(self):
        return [
            ds + "/training/" for ds in sorted(glob(self.IMAGE_PATH + "*"))
            if os.path.isdir(ds)
        ]

    @property
    def VAL_IMAGE_PATH(self):
        return [
            ds + "/validation/" for ds in sorted(glob(self.IMAGE_PATH + "*"))
            if os.path.isdir(ds)
        ]

    @property
    def TEST_IMAGE_PATH(self):
        return [
            ds + "/testing/" for ds in sorted(glob(self.IMAGE_PATH + "*"))
            if os.path.isdir(ds)
        ]

    @property
    def SAM_CHECKPOINT(self):
        return {
            "vit_b": self.ch_root + "sam_vit_b_01ec64.pth",
            "vit_h": self.ch_root + "sam_vit_h_4b8939.pth",
            "med_sam": self.ch_root + "medsam_vit_b.pth",
        }

    @property
    def ENCODER_TYPE(self):
        return {
            "vit_b": "vit_b",
            "vit_h": "vit_h",
            "med_sam": "vit_b",
            "samri": "vit_b",
        }

def get_checkpoint(path, vitb_ckpt):
    cp_list = sorted(glob(path + "*pth"))
    if len(cp_list) == 0:
        cp_name = vitb_ckpt
        start_epoch = 0
    else:
        cp_names = [(os.path.basename(cp)[:-4]) for cp in cp_list]
        start_epoch = max([int(cp.split('_')[-1]) for cp in cp_names if cp != ""])
        cp_name = glob(path + f"*_{str(start_epoch)}.pth*")[0]
    return cp_name, start_epoch

def _get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) :
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
    resize_long = _get_preprocess_shape(mask.shape[-2], mask.shape[-1], target_size)
    resized_mask = F.interpolate(mask, 
                           resize_long, 
                           mode="nearest")
    # Pad
    h, w = resized_mask.shape[-2:]
    padh = target_size - h
    padw = target_size - w
    x = F.pad(resized_mask, (0, padw, 0, padh))
    return x

class MaskSplit():
    """
    Split the labeled ground truth masks into single binary mask. 

    Args:
        mask (np.darray): the labeled ground truth mask. CHW=(1,255,255)
        
    Returns:
        masks (list): A list of splited masks. HW=(255,255)
        
        labels (list): The list of splited masks labels.
    
    """
    def __init__(self, mask):
        self.mask = mask[0, :, :]
        # mask_number (int): the number of the gt mask labels.
        self.mask_number = len(np.unique(self.mask)) - 1
        # masks (list): the list of single mask with different lables, HW=(255,255)
        self.masks, self.labels = self._split_masks()
        """
        Args:
            mask (np.darray): the labeled ground truth mask. CHW=(1,255,255)
            mask_number (int): the number of the gt mask labels.
            masks (list): the list of single mask with different lables, HW=(255,255)
        """

    def __len__(self):
        """
        The number of labels.
        """
        return self.mask_number
    
    def __getitem__(self, index):
        return self.masks[index], self.labels[index]
    
    def _split_masks(self):
        masks = []
        labels = []
        for label in np.unique(self.mask).nonzero()[0]:
            masks.append(self.mask == np.unique(self.mask)[label])
            labels.append(np.unique(self.mask)[label])
        return masks, labels
    
        
def gen_points(mask, num_points=1):
    """
    Generate a point list [H, W] or points [[H, W], ...] in a mask.

    Parameters:
        mask (np.array): the mask in the shape of HW=(255,255) logit type
        num_points: the number of points will be generated. If the number lager
                    than 1, this function will return to a array listing all the 
                    points tuples in a list.

    Returns:
        (np.array): a [W, H] point List if the num_points = 1;
        OR
        (np.array)[[list], ...]: a list of point lists if the num_points > 1.
    """
    h, w = np.nonzero(mask)
    if num_points == 1:
        p_idx = random.randint(int(len(h)*0.45), int(len(h)*0.55))
        return np.array([[w[p_idx], h[p_idx]]])
    else:
        points = []
        for _ in range(num_points):
            p_idx = random.randint(int(len(h)*0.45), int(len(h)*0.55))
            points.append([w[p_idx], h[p_idx]])
        return np.array(points)

def gen_points_torch(mask, num_points=1):
    """
    Generate a point list [H, W] or points [[H, W], ...] in a mask.

    Parameters:
        mask (np.array): the mask in the shape of HW=(255,255) logit type
        num_points: the number of points will be generated. If the number lager
                    than 1, this function will return to a array listing all the 
                    points tuples in a list.

    Returns:
        (np.array): a [W, H] point List if the num_points = 1;
        OR
        (np.array)[[list], ...]: a list of point lists if the num_points > 1.
    """
    non_zero = torch.nonzero(mask)
    if num_points == 1:
        p_idx = random.randint(int(len(non_zero)*0.45), int(len(non_zero)*0.55))
        return non_zero[p_idx]
    else:
        points = []
        for i in range(num_points):
            p_idx = random.randint(int(len(non_zero)*0.45), int(len(non_zero)*0.55))
            points.append(non_zero[p_idx])
        return torch.stack(points)

def gen_bboxes(mask, num_bboxes=1, jitter=0):
    """
    Generate a bounding box tupple with a shape of [min_w, min_h, max_w, max_h]
    or tupple list of multiple bounding boxes.

    Parameters:
        mask (np.array): the mask in the shape of HW=(255,255) logit type
        num_bboxes(Tupple): the number of bounding boxes will be generated. If 
                    the number lager than 1, this function will return to a array
                    listing all the bounding boxes tupples in a list.
        jitter (int): the random shift of the original bounding box.

    Returns:
        (list): a [min_w, min_h, max_w, max_h] bounding box list if the
                num_bboxes = 1;
        [[list], ...]: a list of bounding box lists if the num_bboxes > 1. 
    """
    h, w = np.nonzero(mask)
    bbox = [np.min(w), np.min(h), np.max(w), np.max(h)]

    if np.max(h) - np.min(h) > jitter + 10:
        bbox[1] = max(0, (np.min(h) + _rand_shift(jitter)))
        bbox[3] = min(mask.shape[0], (np.max(h) + _rand_shift(jitter)))
    if np.max(w) - np.min(w) > jitter + 10:
        bbox[0] = max(0, (np.min(w) + _rand_shift(jitter)))
        bbox[2] = min(mask.shape[1], (np.max(w) + _rand_shift(jitter)))
        
    if num_bboxes == 1:
        return np.array(bbox)
    else:
        bboxes = []
        for _ in range(num_bboxes):
            bboxes.append(bbox)
        return np.array(bboxes)

def gen_bboxes_torch(mask, num_bboxes=1, jitter=0):
    """
    Generate a bounding box tupple with a shape of [min_w, min_h, max_w, max_h]
    or tupple list of multiple bounding boxes.

    Parameters:
        mask (np.array): the mask in the shape of HW=(255,255) logit type
        num_bboxes(Tupple): the number of bounding boxes will be generated. If 
                    the number lager than 1, this function will return to a array
                    listing all the bounding boxes tupples in a list.
        jitter (int): the random shift of the original bounding box.

    Returns:
        (list): a [min_w, min_h, max_w, max_h] bounding box list if the
                num_bboxes = 1;
        [[list], ...]: a list of bounding box lists if the num_bboxes > 1. 
    """
    non_zero = torch.nonzero(mask)
    min_h, min_w = non_zero.min(axis=0).values
    max_h, max_w = non_zero.max(axis=0).values
    bbox = [min_w, min_h, max_w, max_h]

    if max_h - min_h > jitter + 10:
        bbox[1] = max(torch.tensor(0), (min_h + _rand_shift(jitter)))
        bbox[3] = min(mask.shape[0], (max_h + _rand_shift(jitter)))
    if max_w - min_w > jitter + 10:
        bbox[0] = max(torch.tensor(0), (min_w + _rand_shift(jitter)))
        bbox[2] = min(mask.shape[1], (max_w + _rand_shift(jitter)))
    
    bbox = torch.stack(bbox)
    
    if num_bboxes == 1:
        return bbox
    else:
        bboxes = []
        for _ in range(num_bboxes):
            bboxes.append(bbox)
        return torch.stack(bboxes)
    
def _rand_shift(jitter):
    """
    generate a random shift number from -jitter to jitter.

    Parameters:
        jitter(int): the shift number of the bbox.

    Returns:
        (int): a random shift number from -jitter to jitter
    """
    return random.randint(-jitter, jitter)
    
