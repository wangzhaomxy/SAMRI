# -*- coding: utf-8 -*-
"""
Inference of the MRI-SAM on the nifti datasets.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

import torch
from segment_anything import sam_model_registry
from utils.test_parser import parser
import torch.nn.functional as F


args = parser.parse_args()


device = args.device
medsam_model = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
medsam_model = medsam_model.to(device)
medsam_model.eval()