# -*- coding: utf-8 -*-

"""
Permanent variables, universal functions, ect
"""

pre_path = "/scratch/user/s4670484/" # The main folder of files
img_path = "MSK/T2_preprocessed_resample_scale2_crop_cv_7c_slice_axis_0_class_7_thresh_7/"
SAVE_PATH = "/home/s4670484/Documents/"
MODEL_SAVE_PATH = pre_path + "cp_temp/"

TRAIN_IMAGE_PATH = [pre_path + img_path + 'set_1/train/',
                    pre_path + img_path + 'set_2/train/',
                    pre_path + img_path + 'set_3/train/',
                    ]

TEST_IMAGE_PATH = [pre_path + img_path + 'set_1/test/',
                    pre_path + img_path + 'set_2/test/',
                    pre_path + img_path + 'set_3/test/',
                    ]

IMAGE_KEYS = "*T2_img*"  # The image file names containing letters between *
MASK_KEYS = "*A_seg*"   # The mask file names containing letters between *

ENCODER_TYPE = {"vit_b":"vit_b",
                  "vit_h":"vit_h",
                  "med_sam":"vit_b",
                  "samri":"vit_b"
                  }

SAM_CHECKPOINT = {"vit_b": pre_path + "Model_dir/sam_vit_b_01ec64.pth",
                  "vit_h": pre_path + "Model_dir/sam_vit_h_4b8939.pth",
                  "med_sam": pre_path + "Model_dir/medsam_vit_b.pth",
                  "samri": pre_path + "Model_dir/samri_vitb.pth"
                  }


DEVICE = "cuda"
BATCH_SIZE = 16
NUM_EPOCHS = 50


NUM_POINTS = 3
NUM_BBOXES = 1
JITTER = 10



LABEL_DICTIONARY = {1:"Femur", 2:"Articular Cartilage-F", 3:"Tibia", 4:"Articular Cartilage-T", 5:"Patella", 6:"Articular Cartilage-P"}
LABEL_LIST = ["Femur", "Articular Cartilage-F", "Tibia", "Articular Cartilage-T", "Patella", "Articular Cartilage-P"]


