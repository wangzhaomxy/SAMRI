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
        (np.ndarray): The ground truth mask with the shape of original masks
                    with shape of (1, 256, 256) and range of int[1,6]
    """
    def __init__(self, data_root, label_num=6):
        super().__init__(data_root)
        self.label_num = label_num


    def __getitem__(self, index):
        """
        Read the nifti MRI data and propress the data into np.ndarray type in
        CHW(256, 256, 1) format, with pixel values in [0,1].

        Arguments:
            index (int): the index of the dataset.

        Return:
            (np.ndarray): The input image with the shape of original image in
                        HWC unit8 format, with pixel values in [0, 255]
            (np.ndarray): The ground truth mask with the shape of original masks
                        with shape of (1, 256, 256) and range of int[1,6]
        """
        # load input image and corresponding mask
        nii_img = self._load_nii(self.img_file[index])
        nii_seg = self._load_nii(self.gt_file[index])
        self.cur_name = self.img_file[index]
        
        # preprocess the image to the range of [0,1], CHW=(1, 256 ,256)
        nii_img = self._preprocess(nii_img)
        nii_seg = self._tensor_mask(nii_seg, self.label_num)

        # shape of nii_img is (1, 256, 256), nii_seg is (6, 256, 256)
        return (nii_img, nii_seg)
    

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
    
    def _tensor_mask(self, mask, label_num):
        """
        slice the 1 channel 6 labeled mask into 6 chanel and each channel
        include 1 labeled binary mask.

        Parameters:
            musk (np.darray): The input numpy musk, (C, H, W) = (1, 256, 256).
            label_num(int): The number of labels.

        Returns:
            (np.darray): The multiple channels mask with each channel including
                        1 labeled binary mask. (C, H, W) = (6, 256, 256).
        """
        masks = []
        for i in range(1, label_num + 1):
            masks.append(mask == i)
        return np.array(masks)
