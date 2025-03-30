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
from utils.losses import dice_similarity
from skimage import transform


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

def get_dice_from_ds(model, test_dataset, med_sam=False):
    """
    Get point prompt 

    Args:
        model (SAM model): The SAM model loaded from Checkpoint.
        test_dataset (Dataset): The pytorch dataset from torch.Dataset.ß

    Returns:
        ([p_record], [b_record]): 
            p_record:A list of DSC of point prompt for the test dataset.
            b_record:A list of DSC of bbox prompt for the test dataset.
    """
    predictor = SamPredictor(model)
    p_record = []
    b_record = []

    for image, mask in tqdm(test_dataset):
        # Image embedding inference
        if med_sam:
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

        for each_mask in masks:
            # generate prompts
            point = gen_points(each_mask)
            point_label = np.array([1])
            bbox = gen_bboxes(each_mask, jitter=0)

            # generate mask
            pre_mask_p, _, _ = predictor.predict(
                                point_coords=point,
                                point_labels=point_label,
                                multimask_output=False,
                                #med_sam=med_sam
                            )
            
            pre_mask_b, _, _ = predictor.predict(
                                point_coords=None,
                                point_labels=None,
                                box=bbox[None, :],
                                multimask_output=False,
                                #med_sam=med_sam
                            )

            # save DSC
            p_record.append(dice_similarity(each_mask, pre_mask_p[0, :, :]))
            b_record.append(dice_similarity(each_mask, pre_mask_b[0, :, :]))
    return p_record, b_record

