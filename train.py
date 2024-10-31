# -*- coding: utf-8 -*-
"""
train the mask decoder
freeze image encoder and prompt encoder
"""

import os
join = os.path.join
from tqdm import tqdm
import torch
from segment_anything import sam_model_registry
from utils.dataloader import EmbDataset
from torch.utils.data import DataLoader
from monai.losses import DiceLoss
from torchvision.ops import sigmoid_focal_loss
from utils.utils import *
from utils.prompt import *
from model import SAMRI
from train_predictor import TrainSamPredictor
import glob

# setup global parameters
model_type = "samri"
encoder_type = ENCODER_TYPE[model_type] # choose one from vit_b and vit_h.
batch_size = BATCH_SIZE
data_path = TRAIN_IMAGE_PATH
model_save_path = MODEL_SAVE_PATH
device = DEVICE
num_epochs = NUM_EPOCHS
train_image_path = TRAIN_IMAGE_PATH
train_image_path.remove('/scratch/project/samri/Embedding/totalseg_mr/')

def gen_batch(mask, prompt):
    masks = MaskSplit(mask)
    lenth = len(masks)
    for each_mask in masks:
        if prompt == "point":
            each_prompt = gen_points(each_mask)
        if prompt == "bbox":
            each_prompt = gen_bboxes(each_mask)
        yield (each_mask, each_prompt, lenth)

def main():
    sam_checkpoint, start_epoch = get_checkpoint(model_save_path)
    sam_model = sam_model_registry[encoder_type](sam_checkpoint)
    samri_model = SAMRI(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)
    train_predictor = TrainSamPredictor(samri_model)

    optimizer = torch.optim.AdamW(
        samri_model.mask_decoder.parameters(),
        lr=1e-4, 
        weight_decay=0.1
    )

    dice_loss = DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")

    #train
    losses = []
    train_dataset = EmbDataset(train_image_path)
    train_loader = DataLoader(train_dataset)

    cp_names = [(os.path.basename(cp)[:-4]) for cp in sam_checkpoint]
    start_epoch = max([int(cp.split('_')[-1]) for cp in cp_names if cp != ""])
    prompts = ["point", "bbox"]
    for epoch in range(start_epoch, num_epochs):
        
        # training part
        samri_model.train()
        epoch_loss = 0
        for step, (embedding, mask, ori_size) in enumerate(tqdm(train_loader)):
            embedding = embedding.squeeze(0)
            mask = mask.squeeze(0).numpy()
            ori_size = (ori_size[0].numpy()[0], ori_size[1].numpy()[0])
            train_predictor.set_embedding(embedding, ori_size)
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
                        focal_loss = sigmoid_focal_loss(y_pred, sub_mask, alpha=0.25, gamma=2,reduction="mean")
                        loss = dice_loss(y_pred, sub_mask) + 10 * focal_loss
                        
                        loss.backward()
                        
                        optimizer.step()
                        
                        sub_loss += loss.item()
            epoch_loss += sub_loss / (len(prompts)*lenth)

        epoch_loss /= (step+1)
        losses.append(epoch_loss)

        # torch.save(samri_model.state_dict(), join(model_save_path, "samri_latest.pth"))
        
        ## save the latest model
        if (epoch + 1) % 1 == 0:
            print(f"The {epoch+1} / {num_epochs} epochs,  Loss: {epoch_loss}.")
            torch.save(samri_model.state_dict(), join(model_save_path, f"samri_vitb_{str(epoch+1)}.pth"))
            print(f"Checkpoint <samri_vitb_{str(epoch+1)}.pth> has been saved.")
        


if __name__ == "__main__":
    main()