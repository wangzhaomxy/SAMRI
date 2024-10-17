import sys
sys.path.append("..")

import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from utils.utils import *
from utils.dataloader import NiiDataset
from tqdm import tqdm
from model import SAMRI
from image_processing.data_processing_code.processing_utile import create_folders, fname_from_path
from glob import glob

base_path = "/scratch/project/samri/"
img_path = base_path + "Datasets/SAMRI_train_test"
save_path = base_path + "Embedding"

folder_names = [fname_from_path(ds) + "/" for ds in glob(img_path + "/*")]
create_folders(save_path + "/", folder_names)

model_type = 'samri'# Choose one from vit_b, vit_h, samri, and med_sam
encoder_tpye = ENCODER_TYPE[model_type]
checkpoint = SAM_CHECKPOINT[model_type]
device = DEVICE

# regist the SAMRI model.
sam_model = sam_model_registry[encoder_tpye](checkpoint)
samri_model = SAMRI(
    image_encoder=sam_model.image_encoder,
    mask_decoder=sam_model.mask_decoder,
    prompt_encoder=sam_model.prompt_encoder,
).to(device)
samri_model.eval()

def save_embedding(img, mask, img_name, save_path):
    train_predictor = SamPredictor(samri_model)
    train_predictor.set_image(img)
    original_image_size = train_predictor.original_size
    embedding = train_predictor.features
    np.savez_compressed(save_path+"/"+img_name[:-7], img=embedding.cpu(), mask=mask, 
             ori_size=original_image_size)
    
for fo_name in tqdm(folder_names):
    print(f"Processing the {fo_name} dataset...")
    img_folder = [img_path + "/" + fo_name + "training/"]
    dataset = NiiDataset(img_folder, multi_mask= True)
    emb_save_path = save_path + "/" + fo_name
    for data, mask in tqdm(dataset):
        img_name = dataset.get_name()
        save_embedding(img=data, mask=mask, img_name=img_name, save_path=emb_save_path)