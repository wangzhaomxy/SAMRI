# -*- coding: utf-8 -*-
"""
train the mask decoder
freeze image encoder and prompt encoder
"""

import os
join = os.path.join
from tqdm import tqdm
import torch
import numpy as np
from segment_anything import sam_model_registry
from utils.dataloader import EmbDataset
from torch.utils.data import DataLoader
from utils.losses import DiceFocalLoss
from utils.utils import *
from utils.prompt import *
from model import SAMRI
from segment_anything.utils.transforms import ResizeLongestSide

# setup global parameters
model_type = "samri"
encoder_type = ENCODER_TYPE[model_type] # choose one from vit_b and vit_h.
batch_size = BATCH_SIZE
model_save_path = MODEL_SAVE_PATH + "ba_rand/"
device = DEVICE
num_epochs = NUM_EPOCHS
train_image_path = TRAIN_IMAGE_PATH
train_image_path.remove('/scratch/project/samri/Embedding/totalseg_mr/')

def main():
    sam_checkpoint, start_epoch = get_checkpoint(model_save_path)
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
    prompts = ["bbox"] #  ["point", "bbox"]
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
        prompts = ["mixed"] #  ["point", "bbox"]
        for prompt in prompts:
            for step, (embedding, masks, ori_size) in enumerate(tqdm(train_loader)):
                # Train model
                ori_size = [(ori_size[0].numpy()[i], ori_size[1].numpy()[i]) for i in range(len(ori_size[0]))]
                
                step += 1
                if prompt == "point":
                    batch_input = [
                        {'image': image.squeeze(),
                            'point_coords':resize_transform.apply_coords_torch(torch.as_tensor(np.array(gen_points(mask.squeeze(0).numpy())), device=gpu), original_size=(256, 256)),
                            'point_labels':torch.as_tensor([1]),
                            'original_size':ori_size
                            } 
                        for image, mask, ori_size in zip(embedding, masks, ori_size)
                    ]
                if prompt == "bbox":
                    batch_input = [
                        {'image': image.squeeze(),
                            'boxes':resize_transform.apply_boxes_torch(torch.as_tensor(np.array(gen_bboxes(mask.squeeze(0).numpy(),jitter=JITTER)), device=gpu), original_size=(256, 256)),
                            'original_size':ori_size
                            } 
                        for image, mask, ori_size in zip(embedding, masks, ori_size)
                    ]
                if prompt == "mixed":
                    batch_input = [
                        {'image': image.squeeze(),
                            'point_coords':resize_transform.apply_coords_torch(torch.as_tensor(np.array(gen_points(mask.squeeze(0).numpy())), device=gpu), original_size=(256, 256)),
                            'point_labels':torch.as_tensor([1]),
                            'boxes':resize_transform.apply_boxes_torch(torch.as_tensor(np.array(gen_bboxes(mask.squeeze(0).numpy(),jitter=JITTER)), device=gpu), original_size=(256, 256)),
                            'original_size':ori_size
                            } 
                        for image, mask, ori_size in zip(embedding, masks, ori_size)
                    ]

                y_pred = samri_model(batch_input, multimask_output=False, train_mode=True, embedding_inputs=True)
                # monitor the model output
                if torch.isnan(y_pred).any():
                    print(f"[Rank {gpu}] NaN in model output at step {step}")
                    continue
                loss = dice_focal_loss(y_pred, masks.to(gpu))
                # monitor the loss
                if torch.isnan(y_pred).any():
                    print(f"[Rank {gpu}] NaN in model output at step {step}")
                    continue
                
                loss.backward()
                optimizer.step()

                optimizer.zero_grad()
                epoch_loss += loss.item()
        epoch_loss /= step
        losses.append(epoch_loss)

        ## save the latest model
        if (epoch + 1) % 1 == 0:
            print(f"The {epoch+1} / {num_epochs} epochs,  Loss: {epoch_loss}.")
            torch.save(samri_model.state_dict(), join(model_save_path, f"samri_vitb_ba_rand_{str(epoch+1)}.pth"))
            print(f"Checkpoint <samri_vitb_ba_rand_{str(epoch+1)}.pth> has been saved.")
        

if __name__ == "__main__":
    main()
        