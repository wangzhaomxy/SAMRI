# -*- coding: utf-8 -*-

"""
Permanent variables, universal functions, ect
"""
from glob import glob
root_path = "/scratch/project/samri/"
EMBEDDING_PATH = root_path + "Embedding/" # The main folder of datasets
TEST_PATH = root_path + "Datasets/SAMRI_train_test/"
MODEL_SAVE_PATH = root_path + "Model_save/"

TRAIN_IMAGE_PATH = [ds + "/" for ds in sorted(glob(EMBEDDING_PATH + "*"))]

TEST_IMAGE_PATH = [ds + "/testing/" for ds in sorted(glob(TEST_PATH + "*"))]

IMAGE_KEYS = "*img*"  # The image file names containing letters between *
MASK_KEYS = "*seg*"   # The mask file names containing letters between *

ENCODER_TYPE = {"vit_b":"vit_b",
                  "vit_h":"vit_h",
                  "med_sam":"vit_b",
                  "samri":"vit_b"
                  }

ch_root = "/scratch/user/s4670484/Model_dir/"
SAM_CHECKPOINT = {"vit_b": ch_root + "sam_vit_b_01ec64.pth",
                  "vit_h": ch_root + "sam_vit_h_4b8939.pth",
                  "med_sam": ch_root + "medsam_vit_b.pth",
                  "samri": ch_root + "samri_vitb.pth"
                  }


DEVICE = "cuda"
BATCH_SIZE = 29
NUM_EPOCHS = 1000
NUM_EPO_PER_ROUND = 100
PROMPT_LOOPS = 10
NUM_POINTS = 3
NUM_BBOXES = 1
JITTER = 10


LABEL_DICTIONARY = {1:"Femur", 2:"Articular Cartilage-F", 3:"Tibia", 4:"Articular Cartilage-T", 5:"Patella", 6:"Articular Cartilage-P"}
LABEL_LIST = ["Femur", "Articular Cartilage-F", "Tibia", "Articular Cartilage-T", "Patella", "Articular Cartilage-P"]


