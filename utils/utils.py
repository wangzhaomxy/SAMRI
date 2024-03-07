# -*- coding: utf-8 -*-

"""
Permanent variables, universal functions, ect
"""

IMAGE_PATH = "/home/s4670484/Documents/MSK/code_test/"
IMAGE_KEYS = "*T2_img*"  # The image file names containing letters between *
MASK_KEYS = "*A_seg*"   # The mask file names containing letters between *


ENCODER_TYPE = "vit_h"
SAM_CHECKPOINT_H = "/home/s4670484/Documents/Model_dir/sam_vit_h_4b8939.pth"
DEVICE = "cuda"



NUM_POINTS = 1
NUM_BBOXES = 1
JITTER = 0

SAVE_PATH = "/home/s4670484/Documents/result/sam_vit_h/test/"

LABEL_DICTIONARY = {1:""}