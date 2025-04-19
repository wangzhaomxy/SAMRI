import numpy as np
import os
from glob import glob
from utils.losses import dice_similarity, sd_hausdorff_distance, sd_mean_surface_distance

data_path = "/scratch/project/samri/MedSAM_inference/"


npz_data = np.load(self.npz_files[index])