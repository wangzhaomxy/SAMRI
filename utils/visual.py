# -*- coding: utf-8 -*-

"""
visualization functions
reference: https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from segment_anything import SamPredictor
from utils.utils import *
from utils.prompt import *
from utils.losses import dice_similarity, sd_hausdorff_distance, sd_mean_surface_distance
from skimage import transform
from utils.dataloader import NiiDataset
import pickle


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=150):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='.', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='.', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

def get_test_record_from_ds(model, test_dataset, med_sam=False):
    """
    Calculate the dice score for the test dataset using the SAM model.

    Args:
        model (SAM model): The SAM model loaded from Checkpoint.
        test_dataset (Dataset): The pytorch dataset from torch.Dataset.
        med_sam (bool): Use the image preprocessing method from medSAM. Defalt
                        is False.

    Returns:
        (list):A list of record for every data, and each data consists of a 
            dictionary. For example:
            
            [{
            "img_name":img_fullpath,     # image full path. (str)
            "mask_name":mask_fullpath,   # mask full path. (str)
            "labels":labels,             # list of labels. (list)
            "p_dice":p_dice,         # list of DSC of point prompt. (list)
            "b_dice":b_dice,         # list of DSC of bbox prompt. (list)
            "p_hd":p_hd,             # list of HD of point prompt. (list)
            "b_hd":b_hd,             # list of HD of bbox prompt. (list)
            "p_msd":p_msd,           # list of MSD of point prompt. (list)
            "b_msd":b_msd,           # list of MSD of bbox prompt. (list)
            "pixel_count":pixel_count,  # list of pixel count. (list)
            "area_percentage":area_percentage # list of area percentage. (list)
            },
            {
                ......
            }
            ......                
            ]
    """
    predictor = SamPredictor(model)
    final_record = []
    
    for image, mask in tqdm(test_dataset):
        img_fullpath = test_dataset.cur_name
        mask_fullpath = test_dataset.cur_gt_name
        p_dice, p_hd, p_msd = [], [], []
        b_dice, b_hd, b_msd = [], [], []
        pixel_count, area_percentage = [], []
        labels = []
        H, W = mask.shape[-2:]
        total_pixels = H * W
        # Image embedding inference
        if med_sam: # for MedSAM evaluation.
            # copied from MedSAM pre_CT_MR.py file MR data preprocessing
            # lower_bound, upper_bound = np.percentile(
            #     image[image > 0], 0.5
            # ), np.percentile(image[image > 0], 99.5)
            # image = np.clip(image, lower_bound, upper_bound)
            image = (
                (image - np.min(image))
                / (np.max(image) - np.min(image) + 1e-8)
                * 255.0
            )
            image[image == 0] = 0

            image = np.uint8(image)
            
            image = transform.resize(
                image,
                (1024, 1024),
                order=3,
                preserve_range=True,
                mode="constant",
                anti_aliasing=True,
            )

            mask = transform.resize(
                mask.transpose(1,2,0),
                (1024, 1024),
                order=0,
                preserve_range=True,
                mode="constant",
                anti_aliasing=False,
            )
            mask = mask.transpose(2,0,1)
        
        predictor.set_image(image)
        masks = MaskSplit(mask)

        for each_mask, label in masks:
            # generate prompts
            point = gen_points(each_mask)
            point_label = np.array([1])
            bbox = gen_bboxes(each_mask, jitter=0)

            # generate mask
            pre_mask_p, _, _ = predictor.predict(
                                point_coords=point,
                                point_labels=point_label,
                                multimask_output=False,
                                med_sam=med_sam
                            )
            
            pre_mask_b, _, _ = predictor.predict(
                                point_coords=None,
                                point_labels=None,
                                box=bbox[None, :],
                                multimask_output=False,
                                med_sam=med_sam
                            )

            # save DSC
            labels.append(label)
            p_dice.append(dice_similarity(pre_mask_p[0, :, :], each_mask))
            b_dice.append(dice_similarity(pre_mask_b[0, :, :], each_mask))
            p_hd.append(sd_hausdorff_distance(pre_mask_p[0, :, :], each_mask))
            b_hd.append(sd_hausdorff_distance(pre_mask_b[0, :, :], each_mask))
            p_msd.append(sd_mean_surface_distance(pre_mask_p[0, :, :], each_mask))
            b_msd.append(sd_mean_surface_distance(pre_mask_b[0, :, :], each_mask))
            pixel_count.append(np.sum(each_mask))
            area_percentage.append(np.sum(each_mask) / total_pixels) 
        
        single_data_result = {"img_name":img_fullpath,
                              "mask_name":mask_fullpath,
                              "labels":labels,
                              "p_dice":p_dice,
                              "b_dice":b_dice,
                              "p_hd":p_hd,
                              "b_hd":b_hd,
                              "p_msd":p_msd,
                              "b_msd":b_msd,
                              "pixel_count":pixel_count,
                              "area_percentage":area_percentage}
        final_record.append(single_data_result)
    return final_record

def get_pix_num_from_ds(test_dataset):
    """
    Calculate the dice score for the test dataset using the SAM model.

    Args:
        model (SAM model): The SAM model loaded from Checkpoint.
        test_dataset (Dataset): The pytorch dataset from torch.Dataset.
        med_sam (bool): using the data preprocessing method from medSAM. Defalt
                        is False.

    Returns:
        ([p_record], [b_record]): 
            p_record:A list of DSC of point prompt for the test dataset.
            b_record:A list of DSC of bbox prompt for the test dataset.
    """
    
    pixel_count = []
    area_percentage = []
    for _, mask in tqdm(test_dataset):
        H, W = mask.shape[-2:]
        total_pixels = H * W
        masks = MaskSplit(mask)
        for each_mask in masks:
            pixel_count.append(np.sum(each_mask))
            area_percentage.append(np.sum(each_mask) / total_pixels)
    return pixel_count, area_percentage
        
def save_test_record(file_paths, sam_model, save_path, med_sam=False, by_ds=False):
    """Save the test record for the test model and dataset.
    

    Args:
        file_paths (list): The testing dataset path list. Ex.["DS1", "DS2", ...]
        sam_model (SAM model): The SAM model loaded from Checkpoint.
        save_path (str): The path to save the record.
        med_sam (bool, optional): If true, using the data preprocessing method
                                  from medSAM. Defaults to False.
        by_ds (bool, optional): if true, saving the result by dataset. 
                                Defaults to False. 
    """
    final_record = {}
    for file_path in file_paths:
        print("Processing the dataset: ",file_path)
        ds_name = file_path.split("/")[-3]
        test_dataset = NiiDataset([file_path], multi_mask= True)    
        ds_record = get_test_record_from_ds(model=sam_model, 
                                                 test_dataset=test_dataset, 
                                                 med_sam=med_sam)
        final_record[ds_name] = ds_record
        if by_ds:
            make_dir(save_path + "/")
            with open(save_path + "/" + ds_name, "wb") as f:
                pickle.dump(ds_record, f)
    if not by_ds:
        with open(save_path, "wb") as f:
            pickle.dump(final_record, f)
            
def save_pxl_record(file_paths, save_path):
    """Save the mask pixel count result for the test dataset.
    
    Args:
        file_paths (list): The testing dataset path list. Ex.["DS1", "DS2", ...]
        save_path (list): The path to save the record.
    """
    pixel_count, area_percentage = [], []
    for idx, file_path in enumerate(file_paths):
        print(f"{idx+1}/{len(file_paths)} :Processing the dataset: ",file_path)
        test_dataset = NiiDataset([file_path], multi_mask= True)
        pixel_count_vit, area_percentage_vit = get_pix_num_from_ds(test_dataset=test_dataset)
        pixel_count.append(pixel_count_vit)
        area_percentage.append(area_percentage_vit)
        final_record = {"pixel_count":pixel_count,"area_percentage":area_percentage}
    with open(save_path, "wb") as f:
        pickle.dump(final_record, f)
        
def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)