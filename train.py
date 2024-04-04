# -*- coding: utf-8 -*-
"""
train the mask decoder
freeze image encoder and prompt encoder
"""

import os
join = os.path.join
from tqdm import tqdm
import torch
import torch.nn as nn
from segment_anything import sam_model_registry
import torch.nn.functional as F
from datetime import datetime
from utils.dataloader import NiiDataset
import wandb
from utils.utils import *
from utils.losses import *
from utils.prompt import *
from torch.utils.data import random_split
from model import SAMRI


# setup global parameters
encoder_type = ENCODER_TYPE["vit_b"] # choose one from vit_b and vit_h.
checkpoint = SAM_CHECKPOINT[encoder_type]
batch_size = BATCH_SIZE
data_path = TRAIN_IMAGE_PATH
timestamp = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = MODEL_SAVE_PATH
device = DEVICE
num_epochs = NUM_EPOCHS
train_image_path = TRAIN_IMAGE_PATH

wandb.login()
experiment = wandb.init(
    project="SAMRI",
    config={
        "batch_size": batch_size,
        "data_path": data_path,
        "model_type": encoder_type,
    },
)


def gen_batch(image, mask, prompt):
    masks = MaskSplit(mask)
    for each_mask in masks:
        if prompt == "point":
            each_prompt = gen_points(each_mask)
        if prompt == "bbox":
            each_prompt = gen_bboxes(each_mask)
        yield (image, each_mask, each_prompt)

def main():
    sam_model = sam_model_registry[encoder_type](checkpoint=checkpoint)
    samri_model = SAMRI(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)

    samri_model.train()

    optimizer = torch.optim.Adam(
        samri_model.mask_decoder.parameters()
    )

    dice_loss = MultiClassDiceLoss()
    bce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    #train
    iter_num = 0
    losses = []
    best_loss = 1e10
    train_dataset = NiiDataset(train_image_path)
    n_val = int(len(train_dataset) * 0.1)
    n_train = len(train_dataset) - n_val
    train_set, val_set = random_split(train_dataset, [n_train, n_val], 
                                    generator=torch.Generator().manual_seed(0))

    start_epoch = 0
    prompts = ["point", "bbox"]
    for epoch in range(start_epoch, num_epochs):
        # training part
        epoch_loss = 0
        for step, (image, mask) in enumerate(tqdm(train_set)):
            sub_loss = 0
            for prompt in prompts:
                optimizer.zero_grad()
                b_image, b_mask, b_prompt = gen_batch(image, mask, prompt)
                image, mask = image.to(device), mask.to(device)
                if prompt == "point":
                    y_pred = samri_model(b_image, points=b_prompt)
                if prompt == "bbox":
                    y_pred = samri_model(b_image, bbox=b_prompt)
                loss = dice_loss(y_pred, b_mask) + bce_loss(y_pred, mask)
                loss.backward()
                optimizer.step()
                sub_loss += loss.item()
            epoch_loss += sub_loss / len(prompts)
            iter_num += 1

        epoch_loss /= step
        losses.append(epoch_loss)
        wandb.log({"epoch_loss": epoch_loss})
        print(
            f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}'
        )
        ## save the latest model
        checkpoint = {
            "model": samri_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, join(model_save_path, "samri_model_latest.pth"))
        
        # validation part
        val_loss = 0
        for image, mask in val_set:

            pass
            ## save the best model
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                checkpoint = {
                    "model": samri_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                }
                torch.save(checkpoint, join(model_save_path, "mri_sam_model_best.pth"))


if __name__ == "__main__":
    main()