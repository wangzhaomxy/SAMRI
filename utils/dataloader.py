# -*- coding: utf-8 -*-

"""
Data loading and image preprocessing

"""

import numpy as np
import os
join = os.path.join
from torch.utils.data import Dataset
import glob
import nibabel as nib
from utils.utils import IMAGE_KEYS, MASK_KEYS
from skimage import exposure


class NiiDataset(Dataset):
    def __init__(self, data_root):
        super().__init__()
        self.data_root = data_root
        self.img_file = []
        self.gt_file = []
        for path in self.data_root:
            self.img_file += sorted(glob.glob(path + IMAGE_KEYS))
            self.gt_file += sorted(glob.glob(path + MASK_KEYS))
        self.cur_name = ""
        # print(f"number of images: {len(self.img_file)}")
        """
        Args:
            data_root (str): The path of the dataset
            img_file (list): The absolute path of the image files list
            gr_file (list): The absolute pathe of the ground truth masks list.
            cur_name: The current image name that the dataset is loading.
        """

    def __len__(self):
        return len(self.img_file)

    def __getitem__(self, index):
        """
        Read the nifti MRI data and propress the data into np.ndarray type in
        HWC(256, 256, 3) unit8 format, with pixel values in [0, 255], to meet 
        the requirement of the SAM predictor.set_image input.

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
        
        # preprocess the image to np.ndarray type in unit8 format,(256 ,256 ,3)
        nii_img = self._preprocess(nii_img)

        # shape of nii_img is (256, 256, 3), nii_seg is (1, 256, 256)
        return (nii_img, nii_seg)


    def _load_nii(self, nii_file):
        """
        load nifty image, (C, H, W) = (1, 256, 256)

        parameters:
        nii_file(str): The input nifti file path.

        returns:
        (np.ndarray): The numpy format image, (C, H, W) = (1, 256, 256)
        """
        return nib.load(nii_file).get_fdata()
    
    def _preprocess(self, np_image):
        """
        Normalize input image into np.uint8 type [0,255], and extend the input 
        image into 3 channels from 1 channel. (1, 256, 256) -> (256, 256, 3).

        parameters:
        np_image(np.darray): The input image, (C, H, W) = (1, 256, 256).

        returns:
        (np.ndarray): The output image, (C, H, W) = (256, 256, 3), range [0,255]

        """
        # split out the image HxW.
        sig_chann = np_image[0, :, :]
        # convert 1 chanel to 3 chanels and transform into  HxWxC
        np_3c = np.array([sig_chann, sig_chann, sig_chann]).transpose(1,2,0)
        # normalize pixel number into [0,1]
        np_3c = (np_3c - np_3c.min()) / (np_3c.max() - np_3c.min())
        # transform image data into [0, 255] integer type, which is np.uint8
        np_3c = exposure.equalize_adapthist(np_3c,clip_limit=0.05)
        np_3c = np.round(np_3c * 255)
        return np_3c
    
    def get_name(self):
        """
        Get the image name that the iterator is loading.

        Returns:
            (str): the image name.
        """
        return os.path.basename(self.cur_name)
        