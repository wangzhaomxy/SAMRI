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
import cv2
import pickle

class NiiDataset(Dataset):
    def __init__(self, 
                 data_root, 
                 shuffle=False, 
                 multi_mask=False):
        """
        Args:
            data_root (str): The path of the dataset
            shuffle (bool): If shuffle the data
            multi_mask (bool): if Ture, return multi-labeld masks; if false, 
                                return a random mask from masks.
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
        self.matching_img = cv2.imread("/home/s4670484/Documents/SAMRI/matching_img/groceries.jpg")
        self.matching_img = cv2.cvtColor(self.matching_img, cv2.COLOR_BGR2RGB)

        # print(f"number of images: {len(self.img_file)}")
        """
        Args:
            data_root (str): The path of the dataset
            img_file (list): The absolute path of the image files list
            gt_file (list): The absolute pathe of the ground truth masks list.
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
        self.cur_gt_name = self.gt_file[index]
        
        # preprocess the image to np.ndarray type in unit8 format,(256 ,256 ,3)
        nii_img = self._preprocess(nii_img)

        # shape of nii_img is (256, 256, 3), nii_seg is (1, 256, 256)
        if self.multi_mask:
            return (nii_img, nii_seg)
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

        # Clipping image intensity
        # sig_chann = self._clip_img(sig_chann)

        # Use the Fourier filter
        # sig_chann = self._ft_pre(sig_chann, rate=0.05)

        # convert 1 chanel to 3 chanels and transform into  HxWxC
        np_3c = np.array([sig_chann, sig_chann, sig_chann]).transpose(1,2,0)

        # histogram matching
        # np_3c = exposure.match_histograms(np_3c,self.matching_img)

        # normalize pixel number into [0,1]
        np_3c = (np_3c - np_3c.min()) / (np_3c.max() - np_3c.min() + 1e-8)

        # clip image intensity value between the 0.5th to 99.5th percentale.
        # np_3c = exposure.rescale_intensity(np_3c, in_range=(0.005, 0.995))

        # Clipping image intensity
        # np_3c = self._clip_img(np_3c)
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
    
    def _clip_img(self, image, lower_b=0.5, upper_b=99.5):
        """
        Clip the image intensity in the range of (lower_b, upper_b) percentale.
        """
        lower_bound, upper_bound = np.percentile(
                image[image > 0], lower_b
            ), np.percentile(image[image > 0], upper_b)
        image_data_pre = np.clip(image, lower_bound, upper_bound)
        image_data_pre = (
            (image_data_pre - np.min(image_data_pre))
            / (np.max(image_data_pre) - np.min(image_data_pre))
            * 255.0
        )
        image_data_pre[image == 0] = 0
        return image_data_pre
    
    def _ft_pre(self, image, rate=0.05, mode="high"):
        """
        Use fast Fourier Transformer to preprocessing image.
        """
        # Compute the Fourier Transform
        fourier = np.fft.fft2(image)
        f_shift = np.fft.fftshift(fourier)

        # Create a high-pass filter
        rows, cols = image.shape
        crow, ccol = rows//2, cols//2
        filt_h = np.ones((rows, cols), np.uint8)
        r = int(rate * crow)
        filt_h[crow - r:crow + r, ccol - r:ccol + r] = 0

        # Create a low-pass filter
        filt_l = np.zeros((rows, cols), np.uint8)
        lr = crow - r
        lc = ccol - r
        filt_l[crow - lr:crow + lr, ccol - lc:ccol + lc] = 1

        # Choose a high-pass filter or low-pass filter
        if mode == "high":
            filt = filt_h
        elif mode == "low":
            filt = filt_l
        else:
            filt = np.ones((rows, cols), np.uint8)

        # Apply the filter
        f_filt = f_shift * filt
        f_ishift = np.fft.ifftshift(f_filt)

        # Inverse Fourier transformer
        img_filted = np.fft.ifft2(f_ishift)
        img_filted = np.abs(img_filted)

        return img_filted

    
class EmbDataset(Dataset):
    def __init__(self, 
                 data_root, 
                 shuffle=False,
                 random_mask=False,
                 resize_mask=False,
                 mask_size=256):
        """
        Args:
            data_root (list[str]): The path list of the datasets. The path
                        eliments should be in format of "xxx/xxx/xxx/".
            shuffle (bool): If shuffle the data. Default is False.
            random_mask(bool): If True, return a random mask from the multi masks.
            resize_mask(bool): If True, resize the mask into the mask size.
            mask_size(int): The target size of the mask, HW=(256, 256). Default is 256.

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
        mask = npz_data["mask"]==label
        if not mask.any():
            raise ValueError(f"The following file contains the empty mask: {self.cur_name}")

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
