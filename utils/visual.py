# -*- coding: utf-8 -*-

"""
visualization functions
reference: https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
"""

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from segment_anything import SamPredictor
from utils.utils import *
from utils.prompt import *
from utils.losses import dice_similarity, sd_hausdorff_distance, sd_mean_surface_distance
from utils.dataloader import NiiDataset, EmbDataset
import pickle
import os
from skimage import io, transform

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

def get_test_record_from_ds(model, test_dataset):
    """
    Calculate the dice score for the test dataset using the SAM model.

    Args:
        model (SAM model): The SAM model loaded from Checkpoint.
        test_dataset (Dataset): The pytorch dataset from torch.Dataset.

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
    
    for image, mask, img_fullpath, mask_fullpath in tqdm(test_dataset):
        image = image.squeeze(0).detach().cpu().numpy()
        mask = mask.squeeze(0).detach().cpu().numpy()
        p_dice, p_hd, p_msd = [], [], []
        b_dice, b_hd, b_msd = [], [], []
        pixel_count, area_percentage = [], []
        labels = []
        
        # Image embedding inference
        H, W = mask.shape[-2:]
        total_pixels = H * W
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
                            )
            
            pre_mask_b, _, _ = predictor.predict(
                                point_coords=None,
                                point_labels=None,
                                box=bbox[None, :],
                                multimask_output=False,
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
        
def save_test_record(file_paths, sam_model, save_path, by_ds=False):
    """Save the test record for the test model and dataset.
    

    Args:
        file_paths (list): The testing dataset path list. Ex.["DS1", "DS2", ...]
        sam_model (SAM model): The SAM model loaded from Checkpoint.
        save_path (str): The path to save the record.
        by_ds (bool, optional): if true, saving the result by dataset. 
                                Defaults to False. 
    """
    final_record = {}
    for file_path in file_paths:
        print("Processing the dataset: ",file_path)
        ds_name = file_path.split("/")[-3]
        test_dataset = NiiDataset([file_path], 
                                  multi_mask= True, 
                                  with_name=True)
        test_loader = DataLoader(test_dataset, 
                        num_workers=24)
        ds_record = get_test_record_from_ds(model=sam_model, 
                                                 test_dataset=test_loader)
        final_record[ds_name] = ds_record
        if by_ds:
            make_dir(save_path + "/")
            with open(save_path + "/" + ds_name, "wb") as f:
                pickle.dump(ds_record, f)
    if not by_ds:
        with open(save_path, "wb") as f:
            pickle.dump(final_record, f)
    
def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_infer_outputs_from_ds(model, test_dataset, save_path, ds_name):
    """
    Save SAM model inference and ground truth results as image/npz files.
    Args:
        model: SAM model.
        test_dataset: PyTorch Dataset loader.
        save_path: Base path to save outputs.
        ds_name: Dataset name for folder structuring.
    """
    predictor = SamPredictor(model)

    for image, mask, img_fullpath, _ in tqdm(test_dataset):
        image = image.squeeze(0).detach().cpu().numpy()
        mask = mask.squeeze(0).detach().cpu().numpy()
        H, W = mask.shape[-2:]

        predictor.set_image(image)
        
        if isinstance(img_fullpath, (tuple, list)):
            img_fullpath = img_fullpath[0]
        img_name = os.path.basename(img_fullpath).replace(".nii.gz", "")

        for each_mask, label in MaskSplit(mask):
            gt_seg = each_mask
            bbox = gen_bboxes(each_mask, jitter=0)

            pre_mask_b, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=bbox[None, :],
                multimask_output=False,
            )

            pre_seg = pre_mask_b[0, :, :]
            comb_seg = np.concatenate([gt_seg, pre_seg], axis=1) * 255

            print(f"Saving results for {img_name} with label {label}...")
            print(f"Ground Truth shape: {gt_seg.shape}, Predicted shape: {pre_seg.shape}")
            
            # Save results
            ds_dir = os.path.join(save_path, ds_name)
            result_dir = os.path.join(ds_dir, "results")
            comb_dir = os.path.join(ds_dir, "comb")
            make_dir(result_dir)
            make_dir(comb_dir)

            io.imsave(
                os.path.join(comb_dir, f"comb_{img_name}_{label}.png"),
                comb_seg.astype(np.uint8),
                check_contrast=False,
            )

            np.savez_compressed(
                os.path.join(result_dir, f"{img_name}_{label}.npz"),
                img = image[..., 0],
                gt=gt_seg,
                pred=pre_seg,
            )
            
def save_infer_results(file_paths, sam_model, save_path):
    """
    Save inference outputs as images for the test model and datasets.
    Args:
        file_paths (list): List of dataset paths.
        sam_model: Loaded SAM model.
        save_path (str): Path to save results.
    """
    for file_path in file_paths:
        print("Processing the dataset:", file_path)
        ds_name = file_path.split("/")[-3]
        test_dataset = NiiDataset([file_path], multi_mask=True, with_name=True)
        test_loader = DataLoader(test_dataset, num_workers=1)

        save_infer_outputs_from_ds(model=sam_model, test_dataset=test_loader, save_path=save_path, ds_name=ds_name)