import numpy as np
import os
from glob import glob
from utils.losses import (dice_similarity, 
                          sd_hausdorff_distance, 
                          sd_mean_surface_distance)
import pickle
from tqdm import tqdm

def get_test_record_from_ds(test_dataset):
    """
    Calculate the dice score for the test dataset using the SAM model.

    Args:
    
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
    final_record = []
    
    for each_data in tqdm(test_dataset):
        data_dict = np.load(each_data, allow_pickle=True)
        gt = data_dict["gt"]
        medsam_seg = data_dict["medsam"]
        label = float(each_data.split("_")[-1][:-4])
        
        b_dice, b_hd, b_msd = [], [], []
        pixel_count, area_percentage = [], []
        labels = []

        H, W = gt.shape[-2:]
        total_pixels = H * W
        
        # save DSC
        labels.append(label)
        b_dice.append(dice_similarity(medsam_seg, gt))
        b_hd.append(sd_hausdorff_distance(medsam_seg, gt))
        b_msd.append(sd_mean_surface_distance(medsam_seg, gt))
        pixel_count.append(np.sum(gt))
        area_percentage.append(np.sum(gt) / total_pixels)
    
        single_data_result = {"img_name": "NaN",
                                "mask_name": each_data,
                                "labels": labels,
                                "p_dice": "NaN",
                                "b_dice":b_dice,
                                "p_hd": "NaN",
                                "b_hd": b_hd,
                                "p_msd": "NaN",
                                "b_msd": b_msd,
                                "pixel_count":pixel_count,
                                "area_percentage":area_percentage}
        final_record.append(single_data_result)
    
    return final_record

def save_test_record(ds_path,  save_path):
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
    file_paths = sorted(glob(ds_path + "*")) # [DS1, DS2, ...] datasets path.
    for file_path in file_paths:
        print("Processing the dataset: ",file_path)
        ds_name = file_path.split("/")[-1] # str, DS name.
        test_dataset = glob(file_path + "/results/*") # [DS1/img1, DS1/img2, ...] with suffix
        ds_record = get_test_record_from_ds(test_dataset=test_dataset)

        final_record[ds_name] = ds_record

        with open(save_path, "wb") as f:
            pickle.dump(final_record, f)
            
if __name__ == "__main__":
    data_path = "/scratch/project/samri/MedSAM_inference/"
    save_path = "/home/s4670484/Desktop/Scratch/samri/samri/Eval_results/med_sam_rep/med_sam_rep.pkl"
    save_test_record(data_path, save_path)
    print("Done!")
