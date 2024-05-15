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
from datetime import datetime
from utils.dataloader import NiiDataset
from torch.utils.data import DataLoader
import wandb
from monai.losses import DiceLoss
from torchvision.ops import sigmoid_focal_loss
from utils.utils import *
from utils.prompt import *
from model import SAMRI
from segment_anything.utils.transforms import ResizeLongestSide

# setup global parameters
model_type = "vit_b"
encoder_type = ENCODER_TYPE[model_type] # choose one from vit_b and vit_h.
sam_checkpoint = SAM_CHECKPOINT[model_type]
batch_size = BATCH_SIZE
data_path = TRAIN_IMAGE_PATH
model_save_path = MODEL_SAVE_PATH
device = DEVICE
num_epochs = NUM_EPOCHS
train_image_path = TRAIN_IMAGE_PATH
amp = True
wandb.login()
experiment = wandb.init(
    project="SAMRI",
    config={
        "batch_size": batch_size,
        "data_path": data_path,
        "model_type": encoder_type,
    },
)

def main():
    sam_model = sam_model_registry[encoder_type](sam_checkpoint)
    samri_model = SAMRI(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)

    train_dataset = NiiDataset(train_image_path)
    batch_data = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        )
    
    resize_transform = ResizeLongestSide(samri_model.image_encoder.img_size)

    optimizer = torch.optim.AdamW(
        samri_model.mask_decoder.parameters(),
        lr=1e-4, 
        weight_decay=0.1
    )

    dice_loss = DiceLoss(sigmoid=True, squared_pred=True, reduction="mean", batch=True)

    #train
    losses = []
    best_loss = 1e5
    scaler = torch.cuda.amp.GradScaler()
    start_epoch = 0
    prompts = ["point", "bbox"] # ["bbox"] #  
    for epoch in range(start_epoch, num_epochs):
        print(f"The {epoch+1} / {num_epochs} epochs.")
        # training part
        samri_model.train()
        epoch_loss = 0
        step = 0

        for each_batch in tqdm(batch_data):
            # Train model
            for prompt in prompts:
                step += 1                    
                if prompt == "point":
                    batch_input = [
                        {'image': prep_img(image, resize_transform),
                            'point_coords':resize_transform.apply_coords_torch(gen_points_torch(mask.squeeze(0)), original_size=image.shape[:2]),
                            'point_labels':torch.as_tensor([[1]], device=device),
                            'original_size':image.shape[:2]
                            } 
                        for image, mask in zip(each_batch[0], each_batch[1])
                    ]
                if prompt == "bbox":
                    batch_input = [
                        {'image': prep_img(image, resize_transform),
                            'boxes':resize_transform.apply_boxes_torch(gen_bboxes_torch(mask.squeeze(0)), original_size=image.shape[:2]),
                            'original_size':image.shape[:2]
                            } 
                        for image, mask in zip(each_batch[0], each_batch[1])
                    ]
                batch_gt_masks = each_batch[1].float()

                if amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        y_pred = samri_model(batch_input, multimask_output=False, train_mode=True)

                        focal_loss = sigmoid_focal_loss(y_pred, batch_gt_masks, alpha=0.25, gamma=2,reduction="mean")
                        loss = dice_loss(y_pred, batch_gt_masks) + focal_loss

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                else:
                    y_pred = samri_model(batch_input, multimask_output=False, train_mode=True)
                    focal_loss = sigmoid_focal_loss(y_pred, batch_gt_masks, alpha=0.25, gamma=2,reduction="mean")
                    loss = dice_loss(y_pred, batch_gt_masks) + focal_loss
                    loss.backward()
                    optimizer.step()

                optimizer.zero_grad()
                epoch_loss += loss.item()

                experiment.log({"sub_loss": loss.item()})

    epoch_loss /= step
    losses.append(epoch_loss)
    experiment.log({"train_epoch_loss": epoch_loss,
                    "train_losses":losses})
    print(
        f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}'
        )
    
    ## save the best model
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(samri_model.state_dict(), join(model_save_path, "samri_vitb_best_rm.pth"))

    ## save the latest model
    torch.save(samri_model.state_dict(), join(model_save_path, "samri_vitb_latest_rm.pth"))


if __name__ == "__main__":
    main()
        