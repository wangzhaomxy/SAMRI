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
from datetime import datetime
from utils.dataloader import NiiDataset
import wandb
from utils.utils import *
from utils.losses import *
from utils.prompt import *
from torch.utils.data import random_split
from model import SAMRI
from train_predictor import TrainSamPredictor

# setup global parameters
model_type = "vit_b"
encoder_type = ENCODER_TYPE[model_type] # choose one from vit_b and vit_h.
sam_checkpoint = SAM_CHECKPOINT[model_type]
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

def gen_batch(mask, prompt):
    masks = MaskSplit(mask)
    lenth = 0
    for each_mask in masks:
        if prompt == "point":
            each_prompt = gen_points(each_mask)
        if prompt == "bbox":
            each_prompt = gen_bboxes(each_mask)
        lenth += 1
        yield (each_mask, each_prompt, lenth)

def main():
    sam_model = sam_model_registry[encoder_type](sam_checkpoint)
    samri_model = SAMRI(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)
    train_predictor = TrainSamPredictor(samri_model)

    optimizer = torch.optim.Adam(
        samri_model.mask_decoder.parameters()
    )

    dice_loss = DiceLoss()
    bce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    #train
    iter_num = 0
    losses = []
    train_dataset = NiiDataset(train_image_path)

    start_epoch = 0
    prompts = ["point", "bbox"]
    for epoch in range(start_epoch, num_epochs):
        print(f"The {epoch+1} / {num_epochs} epochs.")
        # training part
        samri_model.train()
        epoch_loss = 0
        for step, (image, mask) in enumerate(tqdm(train_dataset)):
            train_predictor.set_image(image)
            sub_loss = 0
            for prompt in prompts:
                    for sub_mask, sub_prompt, lenth in gen_batch(mask, prompt):                        
                        optimizer.zero_grad()

                        if prompt == "point":
                            y_pred, _, _ = train_predictor.predict(
                                                        point_coords=sub_prompt,
                                                        point_labels=[1],
                                                        return_logits=True,
                                                        multimask_output=False)
                        if prompt == "bbox":
                            y_pred, _, _  = train_predictor.predict(
                                                        box=sub_prompt[None, :],
                                                        return_logits=True,
                                                        multimask_output=False)

                        sub_mask = torch.tensor(sub_mask[None,:,:], dtype=torch.float, device=torch.device(device))
                        loss = dice_loss(y_pred, sub_mask) + 20 * bce_loss(y_pred, sub_mask)
                        
                        loss.backward()
                        
                        optimizer.step()
                        
                        sub_loss += loss.item()
                        experiment.log({"train_epoch_loss": epoch_loss})
            epoch_loss += sub_loss / (len(prompts)*lenth)
            iter_num += 1

        epoch_loss /= step
        losses.append(epoch_loss)
        experiment.log({"train_epoch_loss": epoch_loss})
        print(
            f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}'
        )
        ## save the latest model
        torch.save(samri_model.state_dict(), join(model_save_path, "samri_vitb_latest.pth"))
        

if __name__ == "__main__":
    main()