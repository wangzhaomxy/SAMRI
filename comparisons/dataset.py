# -*- coding: utf-8 -*-

"""
Data loading and image preprocessing
"""

from utils.dataloader import NiiDataset
from utils.utils import *
from glob import glob
import numpy as np

class UnetNiiDataset(NiiDataset):
    """
    Read the nifti MRI data .

    Args:
        data_root (str): The path of the dataset
        img_file (list): The absolute path of the image files list
        gr_file (list): The absolute pathe of the ground truth masks list.
        cur_name: The current image name that the dataset is loading.

    Return:
        (np.ndarray): The input image with the shape of original image in
                    CHW format of (1, 256, 256) and range of [0, 1]
        (np.darray): The labeled mask. (C, H, W) = (1, 256, 256).
    """
    def __init__(self, data_root):
        super().__init__(data_root)
    
    def _preprocess(self, np_image):
        """
        Normalize input image into range [0,1] with the shape of CHW = 
        (1, 256, 256).

        parameters:
        np_image(np.darray): The input numpy image, (C, H, W) = (1, 256, 256).

        returns:
        (np.ndarray): The output image, (C, H, W) = (1, 256, 256), range [0,1]

        """
        # normalize pixel number into [0,1]

        return (np_image - np_image.min()) / (np_image.max() - np_image.min())
    