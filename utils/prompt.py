# -*- coding: utf-8 -*-
"""
Split ground truth masks. Generate points and bonding boxes.
"""

import numpy as np
import random

class MaskSplit():
    """
    Split the labeled ground truth masks into single logits mask. 
    """
    def __init__(self, mask):
        self.mask = mask[0, :, :]
        self.mask_number = np.max(self.mask)
        self.masks = self._split_masks(self.mask)
        """
        Args:
            mask (np.darray): the labeled ground truth mask. CHW=(1,255,255)
            mask_number (int): the number of the gt mask labels.
            masks (list): the list of single mask with different lables, HW=(255,255)
        """

    def __len__(self):
        return self.mask_number
    
    def __getitem__(self, index):
        return self.masks[index]
    
    def _split_masks(self, mask):
        masks = []
        for i in range(1, self.mask_number+1):
            masks.append(self.mask == i)
        return masks
    
    
def gen_points(mask, num_points=1):
    """
    Generate a point tupple (H, W) or points [(H, W), ...] in a mask.

    Parameters:
        mask (np.array): the mask in the shape of HW=(255,255) logit type
        num_points: the number of points will be generated. If the number lager
                    than 1, this function will return to a array listing all the 
                    points tuples in a list.

    Returns:
        (Tuple): a (H, W) point Tupple if the num_points = 1;
        OR
        [(Tuple), ...]: a list of point Tupples if the num_points > 1.
    """
    h, w = np.nonzero(mask)
    if num_points == 1:
        p_idx = random.randrange(len(h))
        return (h[p_idx], w[p_idx])
    else:
        points = []
        for i in range(num_points):
            p_idx = random.randrange(len(h))
            points.append((h[p_idx], w[p_idx]))
        return points


def gen_bboxes(mask, num_bboxes=1, jitter=0):
    """
    Generate a bounding box tupple with a shape of (min_h, min_w, max_h, max_w)
    or tupple list of multiple bounding boxes.

    Parameters:
        mask (np.array): the mask in the shape of HW=(255,255) logit type
        num_bboxes(Tupple): the number of bounding boxes will be generated. If 
                    the number lager than 1, this function will return to a array
                    listing all the bounding boxes tupples in a list.
        jitter (int): the random shift of the original bounding box.

    Returns:
        (Tuple): a (min_h, min_w, max_h, max_w) bounding box Tupple if the
                num_bboxes = 1;
        [(Tuple), ...]: a list of bounding box Tupples if the num_bboxes > 1. 
    """
    h, w = np.nonzero(mask)
    bbox = (max(0, (h[0] + rand_shift(jitter))),
                 max(0, (w[0] + rand_shift(jitter))), 
                 min(mask.shape[0], (h[-1] + rand_shift(jitter))), 
                 min(mask.shape[1], (w[-1] + rand_shift(jitter))))
    if num_bboxes == 1:
        return bbox
    else:
        bboxes = []
        for _ in range(num_bboxes):
            bboxes.append(bbox)
        return bboxes

def rand_shift(jitter):
    """
    generate a random shift number from -jitter to jitter.

    Parameters:
        jitter(int): the shift number of the bbox.

    Returns:
        (int): a random shift number from -jitter to jitter
    """
    return random.randint(-jitter, jitter)
    
        