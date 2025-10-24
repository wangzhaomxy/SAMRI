# -*- coding: utf-8 -*-
"""
Train the mask decoder (single GPU)
Freeze image encoder and prompt encoder.
Example command:
python train_single_gpu.py \
  --model_type samri \
  --batch_size 48 \
  --data_path ./user_data \
  --model_save_path ./user_data/Model_save \
  --num-epochs 120 \
  --device cuda \
  --save-every 2 \
  --prompts mixed \
"""

# --- Standard library ---
import os
from os.path import join
import argparse

# --- Third-party ---
import torch
import numpy as np
from tqdm import tqdm
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import DataLoader

# --- Local project ---
from model import SAMRI
from utils.dataloader import EmbDataset
from utils.losses import DiceFocalLoss
from utils.utils import *  # contains SAMRIConfig, helper functions, etc.

# setup global parameters (CLI-driven)
cfg = SAMRIConfig()
_parser = argparse.ArgumentParser(add_help=True)
_parser.add_argument("--model-type", "--model_type",
                    dest="model_type",
                    default="samri",
                    choices=list(cfg.ENCODER_TYPE.keys()),
                    help="Model key used to derive encoder_type from ENCODER_TYPE.")
_parser.add_argument("--batch-size", "--batch_size",
                    dest="batch_size",
                    type=int,
                    default=cfg.BATCH_SIZE)
_parser.add_argument("--data-path", "--data_path",
                    dest="data_path",
                    type=str,
                    default=cfg.root_path,
                    help="Root path for datasets and user data.")
_parser.add_argument("--model-save-path", "--model_save_path",
                    dest="model_save_path",
                    type=str,
                    default=cfg.MODEL_SAVE_PATH,
                    help="Directory to save trained models.")
_parser.add_argument("--num-epochs", "--num_epochs",
                    dest="num_epochs",
                    type=int,
                    default=cfg.NUM_EPOCHS,
                    help="Number of training epochs.")
_parser.add_argument("--device",
                    dest="device",
                    type=str,
                    default=cfg.DEVICE,
                    choices=["cuda", "cpu", "mps"],
                    help="Compute device.")
_parser.add_argument("--save-every", type=int, default=1, dest="save_every",
                     help="Save the model every N epochs.")
_parser.add_argument("--prompts", nargs="+", default=["mixed"], dest="prompts",
                     choices=["point","bbox","mixed"],
                     help = "Prompt types for training, choose from 'point', 'bbox', and 'mixed'. 'mixed' means both point and bbox prompts.")
_args, _unknown = _parser.parse_known_args()

# Apply CLI to config + derive globals used below
cfg.root_path = _args.data_path          # triggers dynamic paths in SAMRIConfig (if using the dynamic version)
model_type = _args.model_type
encoder_type = cfg.ENCODER_TYPE[model_type] # choose one from vit_b and vit_h.
batch_size = _args.batch_size
model_save_path = _args.model_save_path
device = _args.device
num_epochs = _args.num_epochs
save_every = _args.save_every
prompts = _args.prompts

# ensure output dir exists
os.makedirs(model_save_path, exist_ok=True)

# Train on EMBEDDINGS (EmbDataset) to match multi-GPU flow
train_image_path = cfg.TRAIN_EMBEDDING_PATH

def main():
    sam_checkpoint, start_epoch = get_checkpoint(model_save_path, cfg.SAM_CHECKPOINT["vit_b"])
    sam_model = sam_model_registry[encoder_type](sam_checkpoint)
    samri_model = SAMRI(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)
    resize_transform = ResizeLongestSide(samri_model.image_encoder.img_size)

    print(
            "Number of total parameters: ",
            sum(p.numel() for p in samri_model.parameters()),
        )  
    print(
        "Number of trainable parameters: ",
        sum(p.numel() for p in samri_model.parameters() if p.requires_grad),
    )
    print("Number of decoder parameters: ", sum(p.numel() for p in samri_model.mask_decoder.parameters()))

    optimizer = torch.optim.AdamW(
        samri_model.mask_decoder.parameters(),
        lr=1e-5, 
        weight_decay=0.1
    )
    
    dice_focal_loss = DiceFocalLoss(sigmoid=True, 
                                     squared_pred=True,
                                     reduction="mean",
                                     lambda_dice=1,
                                     lambda_focal=10)    
    
    #train
    losses = []
    for epoch in range(start_epoch, num_epochs):
        print(f"The {epoch+1} / {num_epochs} epochs.")
        # training part
        samri_model.train()
        epoch_loss = 0
        step = 0
        train_dataset = EmbDataset(train_image_path, 
                                   random_mask=True, 
                                   resize_mask=True, 
                                   mask_size=256)
        num_workers = 8
        train_loader = DataLoader(train_dataset, 
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  shuffle=True)

        for prompt in prompts:
            for step, (embedding, masks, ori_size) in enumerate(tqdm(train_loader)):
                # Train model
                ori_size = [(ori_size[0].numpy()[i], ori_size[1].numpy()[i]) for i in range(len(ori_size[0]))]
                
                step += 1
                if prompt == "point":
                    batch_input = [
                        {'image': image.squeeze(),
                            'point_coords':resize_transform.apply_coords_torch(torch.as_tensor(np.array(gen_points(mask.squeeze(0).numpy())), device=device), original_size=(256, 256)),
                            'point_labels':torch.as_tensor([1]),
                            'original_size':ori_size
                            } 
                        for image, mask, ori_size in zip(embedding, masks, ori_size)
                    ]
                if prompt == "bbox":
                    batch_input = [
                        {'image': image.squeeze(),
                            'boxes':resize_transform.apply_boxes_torch(torch.as_tensor(np.array(gen_bboxes(mask.squeeze(0).numpy(),jitter=cfg.JITTER)), device=device), original_size=(256, 256)),
                            'original_size':ori_size
                            } 
                        for image, mask, ori_size in zip(embedding, masks, ori_size)
                    ]
                if prompt == "mixed":
                    batch_input = [
                        {'image': image.squeeze(),
                            'point_coords':resize_transform.apply_coords_torch(torch.as_tensor(np.array(gen_points(mask.squeeze(0).numpy())), device=device), original_size=(256, 256)),
                            'point_labels':torch.as_tensor([1]),
                            'boxes':resize_transform.apply_boxes_torch(torch.as_tensor(np.array(gen_bboxes(mask.squeeze(0).numpy(),jitter=cfg.JITTER)), device=device), original_size=(256, 256)),
                            'original_size':ori_size
                            } 
                        for image, mask, ori_size in zip(embedding, masks, ori_size)
                    ]

                y_pred = samri_model(batch_input, multimask_output=False, train_mode=True, embedding_inputs=True)
                # monitor the model output
                if torch.isnan(y_pred).any():
                    print(f"NaN in model output at step {step}")
                    continue
                loss = dice_focal_loss(y_pred, masks.to(device))
                # monitor the loss
                if torch.isnan(y_pred).any():
                    print(f"[NaN in model output at step {step}")
                    continue
                
                loss.backward()
                optimizer.step()

                optimizer.zero_grad()
                epoch_loss += loss.item()
        epoch_loss /= step
        losses.append(epoch_loss)

        ## save model
        if (epoch + 1) % save_every == 0:
            print(f"The {epoch+1} / {num_epochs} epochs,  Loss: {epoch_loss}.")
            torch.save(samri_model.state_dict(), join(model_save_path, f"samri_vitb_{str(epoch+1)}.pth"))
            print(f"Checkpoint <samri_vitb_{str(epoch+1)}.pth> has been saved.")
        

if __name__ == "__main__":
    main()
        