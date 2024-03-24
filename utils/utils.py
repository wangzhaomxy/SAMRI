# -*- coding: utf-8 -*-

"""
Permanent variables, universal functions, ect
"""

TRAIN_IMAGE_PATH = ['/home/s4670484/Documents/MSK/T2_preprocessed_resample_scale2_crop_cv_7c_slice_axis_0_class_7_thresh_7/set_1/train/',
                    '/home/s4670484/Documents/MSK/T2_preprocessed_resample_scale2_crop_cv_7c_slice_axis_0_class_7_thresh_7/set_2/train/',
                    '/home/s4670484/Documents/MSK/T2_preprocessed_resample_scale2_crop_cv_7c_slice_axis_0_class_7_thresh_7/set_3/train/',
                    ]
    #"/home/s4670484/Documents/MSK/code_test/"

TEST__IMAGE_PATH = ['/home/s4670484/Documents/MSK/T2_preprocessed_resample_scale2_crop_cv_7c_slice_axis_0_class_7_thresh_7/set_1/test/',
                    '/home/s4670484/Documents/MSK/T2_preprocessed_resample_scale2_crop_cv_7c_slice_axis_0_class_7_thresh_7/set_2/test/',
                    '/home/s4670484/Documents/MSK/T2_preprocessed_resample_scale2_crop_cv_7c_slice_axis_0_class_7_thresh_7/set_3/test/',
                    ]
IMAGE_KEYS = "*T2_img*"  # The image file names containing letters between *
MASK_KEYS = "*A_seg*"   # The mask file names containing letters between *


ENCODER_TYPE = "vit_b"
    # "vit_h"
    # "vit_b"
SAM_CHECKPOINT = "/home/s4670484/Documents/Model_dir/medsam_vit_b.pth"
    # "/home/s4670484/Documents/Model_dir/sam_vit_h_4b8939.pth"
    # "/home/s4670484/Documents/Model_dir/sam_vit_b_01ec64.pth"
    # "/home/s4670484/Documents/Model_dir/medsam_vit_b.pth"
DEVICE = "cuda"

NUM_POINTS = 1
NUM_BBOXES = 1
JITTER = 10

SAVE_PATH = "/home/s4670484/Documents/result/med_sam/test/"
MODEL_SAVE_PATH = "/home/s4670484/Documents/cp_temp/"

LABEL_DICTIONARY = {1:"Femur", 2:"Articular Cartilage-F", 3:"Tibia", 4:"Articular Cartilage-T", 5:"Patella", 6:"Articular Cartilage-P"}
LABEL_LIST = ["Femur", "Articular Cartilage-F", "Tibia", "Articular Cartilage-T", "Patella", "Articular Cartilage-P"]