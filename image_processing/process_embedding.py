# -*- coding: utf-8 -*-
"""
Process and save image embeddings using SAM_vitb model.

Usage example:
python preprocess/precompute_embeddings.py \
  --base-path ./user_data \
  --dataset-path ./user_data/Datasets/SAMRI_train_test/ \
  --img-sub-path train/ \
  --save-path ./user_data/Datasets/Embedding_train/ \
  --checkpoint ./user_data/pretrained_ckpt/sam_vit_b_01ec64.pth \
  --device cuda
"""

import os
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from utils.dataloader import NiiDataset
from tqdm import tqdm
from model import SAMRI
from glob import glob
from utils.utils import SAMRIConfig
import argparse

# ---- CLI ----
_parser = argparse.ArgumentParser(description="Configure SAMRI constants from CLI")
cfg = SAMRIConfig() # default configurations

_parser.add_argument(
    "--base-path",
    default=cfg.root_path,
    help=f"Base project directory (default: {cfg.root_path})",
)
_parser.add_argument(
    "--dataset-path","dataset_path",
    dest="dataset_path",
    default=cfg.IMAGE_PATH,
    help=f"Dataset directory.",
)
_parser.add_argument(
    "--img-sub-path", "img_sub_path",
    dest="img_sub_path",
    default="train",
    choices=["train", "validation", "test"],
    help="Dataset subfolder (default: train)",
)
_parser.add_argument(
    "--save-path","save_path",
    dest="save_path",
    default=cfg.root_path + "Datasets/Embedding_"+_parser.parse_args().img_sub_path+"/",
    help=f"Embedding save directory.",
)
_parser.add_argument(
    "--checkpoint",
    default=cfg.SAM_CHECKPOINT["vit_b"],
    help="Path to SAM_vitb model checkpoint.",
)
_parser.add_argument(
    "--device",
    default=cfg.DEVICE,
    choices=["cuda", "cpu", "mps"],
    help=f"Computation device (default from utils: {cfg.DEVICE})",
)
_args = _parser.parse_args()

cfg.root_path = _args.data_path
img_path = _args.dataset_path
save_path = _args.save_path
img_sub_path = "training/" if _args.img_sub_path == "train" else ("validation/" 
                        if _args.img_sub_path == "validation" else "testing/")    
checkpoint = _args.checkpoint
device = _args.device

# regist the SAMRI model.
sam_model = sam_model_registry["vit_b"](checkpoint)
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
    np.savez_compressed(save_path+"/"+img_name[:-7], img=embedding.detach().cpu(), mask=mask, 
             ori_size=original_image_size)

ds_names = [os.path.basename(ds) + "/" for ds in sorted(glob(img_path + "/*"))]    
for fo_name in tqdm(ds_names):
    print(f"Processing the {fo_name} dataset...")
    img_folder = [img_path + "/" + fo_name + img_sub_path]
    dataset = NiiDataset(img_folder, multi_mask= True)
    emb_save_path = save_path + "/" + fo_name
    os.makedirs(emb_save_path, exist_ok=True)
    for data, mask in tqdm(dataset):
        img_name = dataset.get_name()
        save_embedding(img=data, mask=mask, img_name=img_name, save_path=emb_save_path)