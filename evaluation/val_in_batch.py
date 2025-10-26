# -*- coding: utf-8 -*-
"""
Single-GPU validation script for SAMRI mask decoder.
Freezes image encoder and prompt encoder.

Examples:
    python evaluation.val_in_batch.py \
    --val-emb-path /path/to/val/embeddings/ \
    --ckpt-path /path/to/checkpoint_directory/
    --prompts mixed \
    --device cuda \
    --batch-size 64
    
The results will be saved in a CSV file under the checkpoint directory.
"""

import os
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
from segment_anything import sam_model_registry
from utils.dataloader import EmbDataset
from torch.utils.data import DataLoader
from utils.losses import DiceLoss
from utils.utils import *
from model import SAMRI
from segment_anything.utils.transforms import ResizeLongestSide
import argparse

cfg = SAMRIConfig()
# setup global parameters and converted to CLI-driven.
_parser = argparse.ArgumentParser(add_help=True)

_parser.add_argument("--val-emb-path", "--val_emb_path",
                     dest="val_emb_path",
                     type=str, 
                     default=cfg.root_path + "/Datasets/Embedding_train/",
                     help="The root path of the images.")
_parser.add_argument("--ckpt-path", "--ckpt_path",
                     dest="ckpt_path",
                     type=str, 
                     default=cfg.MODEL_SAVE_PATH,
                     help="The root path or path of the test checkpoint.")
_parser.add_argument("--prompts", nargs="+", default=["mixed"], dest="prompts",
                     choices=["point","bbox","mixed"],
                     help = "Prompt types for training, choose from 'point', 'bbox', and 'mixed'. 'mixed' means both point and bbox prompts.")
_parser.add_argument("--device",
                     dest="device",
                     type=str,
                     default=cfg.DEVICE,
                     help="Device to run the model on.")
_parser.add_argument("--batch-size", "--batch_size",
                     dest="batch_size",
                     type=int,
                     default=cfg.BATCH_SIZE,
                     help="Batch size for validation.")
_args = _parser.parse_args()

# Define global arguments
file_name = "eval_results.csv"  # Example file name, adjust as needed
prompt_mode = _args.prompts()
model_type = "samri"
encoder_type = "vit_b"  # choose one from vit_b and vit_h.
batch_size = _args.batch_size  # Adjust batch size as needed
model_path = _args.ckpt_path
val_emb_path = [ds + "/" for ds in sorted(glob(_args.val_emb_path + "*"))]
device = _args.device

def get_epoch_num(filename):
    match = filename.split('_')[-1].split('.')[0]
    if match:
        return int(match)
    return None

def get_epoch_list_from_df(df_path):
    """
    Extract epoch numbers from a DataFrame saved as a CSV file.
    """
    if os.path.exists(df_path):
        df = pd.read_csv(df_path)
        if 'Epoch' in df.columns:
            return df['Epoch'].tolist()
        else:
            raise ValueError("The DataFrame does not contain an 'Epoch' column.")
    else:
        return []

# Global settings
result_path = os.path.join(model_path, "validation_results")
os.makedirs(result_path, exist_ok=True)
result_save_path = os.path.join(result_path, file_name)
model_files = sorted([f for f in os.listdir(model_path) if 
                                    f.startswith("samri_vitb_box_") and 
                                    get_epoch_num(f) not in 
                                    get_epoch_list_from_df(result_save_path)])

def main():
    print("Batch size:", batch_size)
    print("Model path:", model_path)

    val_dataset = EmbDataset(val_emb_path, resize_mask=True, mask_size=256)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    dice_loss = DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")

    losses = []
    for model_file in model_files:
        epoch = get_epoch_num(model_file)
        print(f"Validating model: {model_file} (Epoch {epoch})")

        sam_model = sam_model_registry[encoder_type](model_path + model_file)
        samri_model = SAMRI(
            image_encoder=sam_model.image_encoder,
            mask_decoder=sam_model.mask_decoder,
            prompt_encoder=sam_model.prompt_encoder,
        ).to(device)

        resize_transform = ResizeLongestSide(samri_model.image_encoder.img_size)
        samri_model.eval()

        loss_list = []
        with torch.no_grad():
            for embeddings, masks, ori_size in tqdm(val_loader, desc=f"Epoch {epoch}"):
                ori_size = [(ori_size[0][i].item(), ori_size[1][i].item()) for i in range(len(ori_size[0]))]
                
                if prompt_mode == "box":
                    batch_input = [
                        {
                            'image': emb.squeeze().to(device),
                            'boxes': resize_transform.apply_boxes_torch(
                                torch.as_tensor(np.array(gen_bboxes(mask.squeeze(0).numpy(), jitter=0)), device=device),
                                original_size=(256, 256)),
                            'original_size': size,
                        }
                        for emb, mask, size in zip(embeddings, masks, ori_size)
                    ]
                elif prompt_mode == "bp":
                    batch_input = [
                        {'image': emb.squeeze().to(device),
                            'point_coords':resize_transform.apply_coords_torch(
                                torch.as_tensor(np.array(gen_points(mask.squeeze(0).numpy())), device=device), 
                                original_size=(256, 256)),
                            'point_labels':torch.as_tensor([1]),
                            'boxes': resize_transform.apply_boxes_torch(
                                torch.as_tensor(np.array(gen_bboxes(mask.squeeze(0).numpy(), jitter=0)), device=device), 
                                original_size=(256, 256)),
                            'original_size':size,
                            } 
                        for emb, mask, size in zip(embeddings, masks, ori_size)
                    ]
                else:
                    raise ValueError("Invalid prompt mode. Choose either 'box' or 'bp'.")
                preds = samri_model(batch_input, multimask_output=False, train_mode=True, embedding_inputs=True)
                loss = dice_loss(preds, masks.to(device))
                loss_list.append(loss.item())

        avg_loss = np.mean(loss_list)
        print(f"Epoch {epoch}: Mean Loss = {avg_loss:.4f}")
        losses.append((epoch, avg_loss))

    df = pd.DataFrame(losses, columns=["Epoch", "DiceScore"])
    if os.path.exists(result_save_path):
        df0 = pd.read_csv(result_save_path)
        df = pd.concat([df0, df], ignore_index=True)
    df.set_index("Epoch", inplace=True)
    df.sort_index(inplace=True)
    df.to_csv(result_save_path)
    print(f"Validation results saved to {result_path}")

if __name__ == "__main__":
    main()
