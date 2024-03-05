
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

# load args from command line
args = test_parser.parse_args()

device = args.device
checkpoint=args.checkpoint
encoder_tpye = args.encoder_tpye

# regist the MRI-SAM model
mri_sam_model = sam_model_registry[encoder_tpye](checkpoint)
mri_sam_model = mri_sam_model.to(device)
predictor = SamPredictor(mri_sam_model)

# load dataset
file_path = IMAGE_PATH
test_dataset = NiiDataset(file_path)

for image, mask in test_dataset:
    # Image embedding inference
    predictor.set_image(image)

    masks = MaskSplit(mask)

