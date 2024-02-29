# -*- coding: utf-8 -*-
"""
Inference of the MRI-SAM on the nifti datasets.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

import torch
from segment_anything import sam_model_registry
from utils.parsers import test_parser
from utils.visual import *
import torch.nn.functional as F

# load args from command line
args = test_parser.parse_args()

device = args.device
checkpoint=args.checkpoint
encoder_tpye = args.encoder_tpye

# regist the MRI-SAM model
mri_sam_model = sam_model_registry[encoder_tpye](checkpoint)
mri_sam_model = mri_sam_model.to(device)
mri_sam_model.eval()



