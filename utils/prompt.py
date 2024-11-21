# -*- coding: utf-8 -*-
"""
Split ground truth masks. Generate points and bonding boxes.
"""

import numpy as np
import torch
import random

class MaskSplit():
    """
    Split the labeled ground truth masks into single logits mask. 

    Args:
        mask (np.darray): the labeled ground truth mask. CHW=(1,255,255)
        
    
    """
    def __init__(self, mask):
        self.mask = mask[0, :, :]
        # mask_number (int): the number of the gt mask labels.
        self.mask_number = len(np.unique(self.mask))
        # masks (list): the list of single mask with different lables, HW=(255,255)
        self.masks = self._split_masks()
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
        return self.masks[index]
    
    def _split_masks(self):
        masks = []
        for label in np.unique(self.mask):
            if label != 0:
                masks.append(self.mask == label)
        return masks
    
    
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
        for i in range(num_points):
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

    if np.max(h) - np.min(h) > 30:
        bbox[1] = max(0, (np.min(h) + rand_shift(jitter)))
        bbox[3] = min(mask.shape[0], (np.max(h) + rand_shift(jitter)))
    if np.max(w) - np.min(w) > 30:
        bbox[0] = max(0, (np.min(w) + rand_shift(jitter)))
        bbox[2] = min(mask.shape[1], (np.max(w) + rand_shift(jitter)))
        
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

    if max_h - min_h > 30:
        bbox[1] = max(torch.tensor(0), (min_h + rand_shift(jitter)))
        bbox[3] = min(mask.shape[0], (max_h + rand_shift(jitter)))
    if max_w - min_w > 30:
        bbox[0] = max(torch.tensor(0), (min_w + rand_shift(jitter)))
        bbox[2] = min(mask.shape[1], (max_w + rand_shift(jitter)))
    
    bbox = torch.stack(bbox)
    
    if num_bboxes == 1:
        return bbox
    else:
        bboxes = []
        for _ in range(num_bboxes):
            bboxes.append(bbox)
        return torch.stack(bboxes)
    
def rand_shift(jitter):
    """
    generate a random shift number from -jitter to jitter.

    Parameters:
        jitter(int): the shift number of the bbox.

    Returns:
        (int): a random shift number from -jitter to jitter
    """
    return random.randint(-jitter, jitter)
    
