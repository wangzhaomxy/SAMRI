# -*- coding: utf-8 -*-

"""
Permanent variables, universal functions, ect
"""
from glob import glob
import os

root_path = "/scratch/project/samri/"
EMBEDDING_PATH = root_path + "Embedding/" # The main folder of datasets
TEST_PATH = root_path + "Datasets/SAMRI_train_test/"
MODEL_SAVE_PATH = root_path + "Model_save/"
DEVICE = "cuda"
BATCH_SIZE = 29
NUM_EPOCHS = 1000
JITTER = 10

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


def get_checkpoint(path):
    cp_list = sorted(glob(path + "*"))
    print("cp_list", cp_list)
    cp_names = [(os.path.basename(cp)[:-4]) for cp in cp_list]
    print("cp_names", cp_names)
    start_epoch = [int(cp.split('_')[-1]) for cp in cp_names if cp != ""]
    cp_name = glob(path + f"*_{str(start_epoch)}.pth*")[0]
    print(start_epoch, " and ", cp_name)
    return cp_name, start_epoch
