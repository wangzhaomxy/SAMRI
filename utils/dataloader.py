# -*- coding: utf-8 -*-

"""
Data loading and image preprocessing

"""

import os
join = os.path.join
import numpy as np
import torch
from torch.utils.data import Dataset
import glob, random
import nibabel as nib
from utils.utils import IMAGE_KEYS, MASK_KEYS, preprocess_mask
import pickle

class NiiDataset(Dataset):
    def __init__(self, 
                 data_root, 
                 shuffle=False, 
                 multi_mask=False,
                 with_name=False):
        """
        Args:
            data_root (list[str]): The path list of the datasets. The path
                        eliments should be in format of "xxx/xxx/xxx/".
            shuffle (bool): If shuffle the data
            multi_mask (bool): if Ture, return multi-labeld masks; if false, 
                                return a random mask from masks.
            with_name (bool): if True, return the image name and mask name in 
                                the format of (image, mask, image_name, mask_name).
        """
        super().__init__()
        self.data_root = data_root
        self.img_file = []
        self.gt_file = []
        for path in self.data_root:
            self.img_file += sorted(glob.glob(path + IMAGE_KEYS))
            self.gt_file += sorted(glob.glob(path + MASK_KEYS))
        if shuffle:
            self.img_file, self.gt_file = self._shuffle(self.img_file, self.gt_file)
        self.cur_name = ""
        self.cur_gt_name = ""
        self.multi_mask = multi_mask
        self.with_name = with_name

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
        self.cur_gt_name = self.gt_file[index]
        
        # preprocess the image to np.ndarray type in unit8 format,(256 ,256 ,3)
        nii_img = self._preprocess(nii_img)

        # shape of nii_img is (256, 256, 3), nii_seg is (1, 256, 256)
        if self.multi_mask:
            if self.with_name:
                return (nii_img, nii_seg, self.cur_name, self.cur_gt_name)
            else:
                return (nii_img, nii_seg)
        else:
            if self.with_name:
                return (nii_img, nii_seg==np.unique(nii_seg)[random.choice(np.unique(nii_seg).nonzero()[0])], self.cur_name, self.cur_gt_name)
            else:
                return (nii_img, nii_seg==np.unique(nii_seg)[random.choice(np.unique(nii_seg).nonzero()[0])])
        
    def _shuffle(self, data1, data2):
        """
        shuffle images and masks simultainiously.

        Arguments:
            data1 (list): list of images path.
            data2 (list): list of masks path.

        Returns:
            (tuple(list,list)): the tuple of shuffled image path list and masks
                                path list.
        """
        zipped_data = list(zip(data1,data2))
        random.shuffle(zipped_data)
        sd_data1, sd_data2 = zip(*zipped_data)
        return (sd_data1, sd_data2)


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
        (np.ndarray): The output image, (H, W, C) = (256, 256, 3), range [0,255]

        """
        # split out the image HxW.
        sig_chann = np_image[0, :, :]

        # convert 1 chanel to 3 chanels and transform into  HxWxC
        np_3c = np.array([sig_chann, sig_chann, sig_chann]).transpose(1,2,0)

        # normalize pixel number into [0,1]
        np_3c = (np_3c - np_3c.min()) / (np_3c.max() - np_3c.min() + 1e-8)

        # transform image data into [0, 255] integer type, which is np.uint8
        np_3c = np.round(np_3c * 255)
        return np_3c
    
    def get_name(self):
        """
        Get the image name that the iterator is loading.

        Returns:
            (str): the image name.
        """
        return os.path.basename(self.cur_name)
    

class EmbDataset(Dataset):
    def __init__(self, 
                 data_root, 
                 shuffle=False,
                 random_mask=False,
                 resize_mask=False,
                 mask_size=256,
                 with_name=False):
        """
        Args:
            data_root (list[str]): The path list of the datasets. The path
                        eliments should be in format of "xxx/xxx/xxx/".
            shuffle (bool): If shuffle the data. Default is False.
            random_mask(bool): If True, return a random mask from the multi masks.
            resize_mask(bool): If True, resize the mask into the mask size.
            mask_size(int): The target size of the mask, HW=(256, 256). Default is 256.
            with_name (bool): if True, return the image name and mask name in 
                                the format of (embedding, mask, original_size, 
                                emb_name).
        """
        super().__init__()
        self.data_root = data_root
        self.npz_files = []
        for path in self.data_root:
            self.npz_files += sorted(glob.glob(path + "*"))

        if shuffle:
            random.shuffle(self.npz_files)
        self.cur_name = ""
        self.random_mask = random_mask
        self.resize_mask = resize_mask
        self.mask_size = mask_size
        self.with_name = with_name

        """
        Vars:
            data_root (str): The path of the dataset
            npz_files (list): The absolute path of the npz files list
            cur_name: The current image name that the dataset is loading.
        """

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, index):
        """
        Read the npz data. The npz data includes 

        Arguments:
            index (int): the index of the dataset.

        Return:
            (np.ndarray): The image embedding with the shape of (1,256,64,64)
            (np.ndarray): The ground truth mask with the shape of original masks
                        with shape of (1, H, W) and range of labels.
            (tuple): The original size of the image.
        """
        npz_data = np.load(self.npz_files[index])
        self.cur_name = self.npz_files[index]
        mask = npz_data["mask"]

        if self.random_mask:
            mask = mask==np.unique(mask)[random.choice(np.unique(mask).nonzero()[0])]
            if not mask.any():
                raise ValueError(f"After random choice, The following file contains the empty mask: {self.get_name()}")

        if self.resize_mask:
            mask = torch.tensor(mask, dtype=torch.float)[None, :, :, :]
            mask = preprocess_mask(mask, target_size=self.mask_size)
            mask = mask.squeeze(0).numpy()
            if not mask.any():
                raise ValueError(f"After resize, The following file contains the empty mask: {self.get_name()}")
        if self.with_name:
            return (npz_data["img"], mask, tuple(npz_data["ori_size"]), self.cur_name)
        else:
            return (npz_data["img"], mask, tuple(npz_data["ori_size"]))

    def get_name(self):
        """
        Get the image name that the iterator is loading.

        Returns:
            (str): the image name.
        """
        return os.path.basename(self.cur_name)
    

class BalancedEmbDataset(Dataset):
    def __init__(self, 
                 data_root,
                 sub_set = "60_up", 
                 resize_mask=False,
                 mask_size=256):
        """
        Args:
            data_root (str): The path of the balanced dataset file.
            sub_set (str): The subset of the dataset. The options are "60_up" and "60_down".
                            60_up: The subset of the dataset with b_dice > 0.6.
                            60_down: The subset of the dataset with b_dice > 0.2 and <= 0.6.
            resize_mask(bool): If True, resize the mask into the mask size.
            mask_size(int): The target size of the mask, HW=(256, 256). Default is 256.

        """
        super().__init__()
        self.data_root = data_root
        if sub_set == "60_up":
            with open(self.data_root, "rb") as f:
                self.file_list = pickle.load(f)["train_60_up"]
        elif sub_set == "60_down":
            with open(self.data_root, "rb") as f:
                self.file_list = pickle.load(f)["train_60_down"]
        elif sub_set == "all":
            with open(self.data_root, "rb") as f:
                file_lists = pickle.load(f)
                self.file_list = file_lists["train_60_up"] + file_lists["train_60_down"]

        self.cur_name = ""
        self.resize_mask = resize_mask
        self.mask_size = mask_size
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        """
        Read the npz data. The npz data includes 

        Arguments:
            index (int): the index of the dataset.

        Return:
            (np.ndarray): The image embedding with the shape of (1,256,64,64)
            (np.ndarray): The ground truth mask with the shape of original masks
                        with shape of (1, H, W) and range of labels.
            (tuple): The original size of the image.
        """
        npz_file_path = self.file_list[index]["emb_path"]
        label = self.file_list[index]["labels"]
        npz_data = np.load(npz_file_path)
        self.cur_name = npz_file_path
        mask = npz_data["mask"] == int(label)
        if not mask.any():
            raise ValueError(f"The following file contains the empty mask: {self.cur_name}, label: {label}")

        if self.resize_mask:
            mask = torch.tensor(mask, dtype=torch.float)[None, :, :, :]
            mask = preprocess_mask(mask, target_size=self.mask_size)
            mask = mask.squeeze(0).numpy()
            if not mask.any():
                raise ValueError(f"After resize, The following file contains the empty mask: {self.get_name()}")
            
        return (npz_data["img"], mask, tuple(npz_data["ori_size"]))

    def get_name(self):
        """
        Get the image name that the iterator is loading.

        Returns:
            (str): the image name.
        """
        return os.path.basename(self.cur_name)


def emb_name_split(data_root, 
             num_of_subset=2, 
             shuffle=True):
    npz_files = []
    for path in data_root:
        npz_files += sorted(glob.glob(path + "*"))
    
    if shuffle:
        random.shuffle(npz_files)
    
    sub_num = len(npz_files) // num_of_subset
    
    return [npz_files[i*sub_num:(i+1)*sub_num] for i in range(num_of_subset)]
