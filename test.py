# -*- coding: utf-8 -*-
"""
Inference of the MRI-SAM on the nifti datasets.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

import torch
import torch.nn.functional as F
from segment_anything import sam_model_registry, SamPredictor
from utils.parsers import test_parser
from utils.visual import *
from utils.utils import *
from torch.utils.data import DataLoader
from utils.dataloader import NiiDataset
from utils.prompt import *
from tqdm import tqdm
from utils.losses import dice_similarity

"""
# load args from command line
args = test_parser.parse_args()

device = args.device
checkpoint=args.checkpoint
encoder_tpye = args.encoder_tpye

# Use args in the formal publics
"""

encoder_tpye = ENCODER_TYPE
checkpoint = SAM_CHECKPOINT_H
device = DEVICE

# regist the MRI-SAM model and predictor.
mri_sam_model = sam_model_registry[encoder_tpye](checkpoint)
mri_sam_model = mri_sam_model.to(device)
predictor = SamPredictor(mri_sam_model)

# load dataset
file_path = IMAGE_PATH
test_dataset = NiiDataset(file_path)

# setup essential parameters.
num_points = NUM_POINTS
num_bboxes = NUM_BBOXES
jitter = JITTER
record = {}
save_path = SAVE_PATH

for image, mask in tqdm(test_dataset):
    # Image embedding inference
    predictor.set_image(image)
    
    name = test_dataset.get_name()
    # split the multi-labeled mask into single labeled
    # logit masks.
    masks = MaskSplit(mask)

    sub_record = {"p":[], "b":[]}
    p_rec = []
    b_rec = []    
    if num_points == num_bboxes == 1:
        for each_label_mask in masks: # shape is HW=(255, 255)
            # generate prompts
            point = gen_points(each_label_mask)
            point_lable = np.array([1])
            bbox = gen_bboxes(each_label_mask, jitter=jitter)

            # generate mask
            pre_mask_p, _, _ = predictor.predict(
                                point_coords=point,
                                point_labels=point_lable,
                                multimask_output=False,
                            )
            
            pre_mask_b, _, _ = predictor.predict(
                                point_coords=None,
                                point_labels=None,
                                box=bbox[None, :],
                                multimask_output=False,
                            )
            
            p_dice = dice_similarity(each_label_mask, pre_mask_p[0, :, :])
            b_dice = dice_similarity(each_label_mask, pre_mask_b[0, :, :])

            p_rec.append(p_dice)
            b_rec.append(b_dice)
        sub_record["p"].append(p_rec)
        sub_record["b"].append(b_rec)

    else:
        pass

    record[name] = sub_record
    
    



