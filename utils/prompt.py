# -*- coding: utf-8 -*-
"""
Split ground truth masks. Generate points and bonding boxes.
"""

import numpy as np

class MaskOperate():
    """
    
    """
    def __init__(self, mask):
        self.mask = mask
        self.mask_number = self._get_number()
        self.masks = self._split_masks(self.mask)

    def _get_number(self):
        return np.max(self.mask)
    
    def _split_masks(self, mask):
        masks = []
        for i in range(1, self.mask_number+1):
            masks.append(self.mask == i)
        return masks
    
    def get_masks(self):
        return self.masks

        