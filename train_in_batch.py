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
from datetime import datetime
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
data_path = TRAIN_IMAGE_PATH
model_save_path = MODEL_SAVE_PATH + "ba/"
device = DEVICE
num_epochs = NUM_EPOCHS
train_image_path = TRAIN_IMAGE_PATH
train_image_path.remove('/scratch/project/samri/Embedding/totalseg_mr/')

def prep_img(image, tramsform, device=device):
    image = tramsform.apply_image(image)
    image = torch.as_tensor(image, device=device)
    return image.permute(2, 0, 1).contiguous()

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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    dice_focal_loass = DiceFocalLoss(sigmoid=True, 
                                     squared_pred=True,
                                     batch= True, 
                                     reduction="mean",
                                     lambda_dice=1,
                                     lambda_focal=10)

    #train
    losses = []
    prompts =["point", "bbox"] 
    for epoch in range(start_epoch, num_epochs):
        print(f"The {epoch+1} / {num_epochs} epochs.")
        # training part
        samri_model.train()
        epoch_loss = 0
        batch_counter = 0
        remain_data = []
        step = 0
        batch_data = []
        train_dataset = EmbDataset(train_image_path)
        train_loader = DataLoader(train_dataset, shuffle=True)
        for step, (embedding, mask, ori_size) in enumerate(tqdm(train_loader)):
            # Generate batch in multiple mask mode.
            embedding = embedding.squeeze()
            masks = MaskSplit(mask.squeeze(0))
            ori_size = (ori_size[0].numpy()[0], ori_size[1].numpy()[0])
            num_masks = len(masks)
            if num_masks > batch_size:
                raise RuntimeError("Too small batch size. It should be larger than label numbers.")
            batch_counter += num_masks
            if batch_counter < batch_size:
                if not remain_data:
                    batch_data += [(embedding, masks[i], ori_size) for i in range(num_masks)]
                else:
                    batch_data += remain_data
                    batch_data += [(embedding, masks[i], ori_size) for i in range(num_masks)]
                    remain_data = []
            else:
                batch_counter -= batch_size
                batch_data += [(embedding, masks[i], ori_size) for i in range(num_masks - batch_counter)]
                if batch_counter != 0:
                    remain_data = [(embedding, masks[i], ori_size) for i in range(num_masks - batch_counter, num_masks)]
                
                # Train model
                for prompt in prompts:
                    step += 1                    
                    if prompt == "point":
                        batch_input = [
                            {'image': image.to(device),
                             'point_coords':resize_transform.apply_coords_torch(torch.as_tensor(np.array([gen_points(mask.numpy())]), device=device), original_size=ori_size),
                             'point_labels':torch.as_tensor([[1]], device=device),
                             'original_size':ori_size
                             } 
                            for image, mask, ori_size in batch_data
                        ]
                    if prompt == "bbox":
                        batch_input = [
                            {'image': image.to(device),
                             'boxes':resize_transform.apply_boxes_torch(torch.as_tensor(np.array([gen_bboxes(mask.numpy())]), device=device), original_size=ori_size),
                             'original_size':ori_size
                             } 
                            for image, mask, ori_size in batch_data
                        ]
                    
                    y_pred = samri_model(batch_input, multimask_output=False, train_mode=True, embedding_inputs=True)
                    batch_gt_masks = torch.stack([preprocess_mask(torch.tensor(x,dtype=torch.float, device=torch.device(device))[None,None,:,:],target_size=256) for _,x,_ in batch_data], dim=0).squeeze(1)
                    loss = dice_focal_loass(y_pred, batch_gt_masks)
                    loss.backward()
                    optimizer.step()

                    optimizer.zero_grad()
                    epoch_loss += loss.item()
                batch_data = []
        scheduler.step()
        epoch_loss /= step
        losses.append(epoch_loss)

        ## save the latest model
        if (epoch + 1) % 1 == 0:
            print(f"The {epoch+1} / {num_epochs} epochs,  Loss: {epoch_loss}.")
            torch.save(samri_model.state_dict(), join(model_save_path, f"samri_vitb_ba_{str(epoch+1)}.pth"))
            print(f"Checkpoint <samri_vitb_{str(epoch+1)}.pth> has been saved.")

if __name__ == "__main__":
    main()
           