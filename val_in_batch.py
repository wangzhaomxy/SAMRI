# -*- coding: utf-8 -*-
"""
Single-GPU validation script for SAMRI mask decoder.
Freezes image encoder and prompt encoder.
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
from utils.prompt import *
from model import SAMRI
from segment_anything.utils.transforms import ResizeLongestSide

# Define global arguments
model_sub_path = "bp_fullds_balance_up/"
# model_sub_path = "fullds_balance_up_new_loss/"

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

# setup parameters
model_type = "samri"
encoder_type = ENCODER_TYPE[model_type]  # choose one from vit_b and vit_h.
batch_size = 256  # Adjust batch size as needed
model_path = MODEL_SAVE_PATH + model_sub_path
val_emb_path = [VAL_EMBEDDING_PATH[0][:-1] + "_zero/"] # VAL_EMBEDDING_PATH
result_path = os.path.join(model_path, "validation_results")
os.makedirs(result_path, exist_ok=True)
result_save_path = os.path.join(result_path, "dice_loss_results_zero.csv")
model_files = sorted([f for f in os.listdir(model_path) if 
                                    f.startswith("samri_vitb_box_") and 
                                    get_epoch_num(f) not in 
                                    get_epoch_list_from_df(result_save_path)])

def main():
    print("Device:", torch.cuda.get_device_name(0))
    print("Batch size:", batch_size)
    print("Model path:", model_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
                
                # batch_input = [
                #     {
                #         'image': emb.squeeze().to(device),
                #         'boxes': resize_transform.apply_boxes_torch(
                #             torch.as_tensor(np.array(gen_bboxes(mask.squeeze(0).numpy(), jitter=0)), device=device),
                #             original_size=(256, 256)),
                #         'original_size': size,
                #     }
                #     for emb, mask, size in zip(embeddings, masks, ori_size)
                # ]
                
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
