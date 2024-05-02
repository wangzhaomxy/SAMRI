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

wandb.login()
experiment = wandb.init(
    project="SAMRI",
    config={
        "batch_size": batch_size,
        "data_path": data_path,
        "model_type": encoder_type,
    },
)

def prep_img(image, tramsform, device=device):
    image = tramsform.apply_image(image)
    image = torch.as_tensor(image, device=device)
    return image.permute(2, 0, 1).contiguous()


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
    resize_transform = ResizeLongestSide(samri_model.image_encoder.img_size)

    optimizer = torch.optim.Adam(
        samri_model.mask_decoder.parameters()
    )

    dice_loss = DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")

    #train
    losses = []
    best_loss = 1e5
    scaler = torch.cuda.amp.GradScaler()
    start_epoch = 0
    prompts = ["point", "bbox"]
    for epoch in range(start_epoch, num_epochs):
        print(f"The {epoch+1} / {num_epochs} epochs.")
        # training part
        samri_model.train()
        epoch_loss = 0
        batch_counter = 0
        remain_data = []
        step = 0
        batch_data = []
        train_dataset = NiiDataset(train_image_path, shuffle=True, multi_mask=True)
        for image, mask in tqdm(train_dataset):
            # Generate batch in multiple mask mode.
            num_masks = len(mask)
            if num_masks > batch_size:
                raise RuntimeError("Too small batch size. It should be larger than label numbers")
            batch_counter += num_masks
            if batch_counter < batch_size:
                if not remain_data:
                    batch_data += [(image, mask[i]) for i in range(num_masks)]
                else:
                    batch_data += remain_data
                    batch_data += [(image, mask[i]) for i in range(num_masks)]
                    remain_data = []
            else:
                batch_counter -= batch_size
                batch_data += [(image, mask[i]) for i in range(num_masks - batch_counter)]
                if batch_counter != 0:
                    remain_data = [(image, mask[i]) for i in range(num_masks - batch_counter, num_masks)]
                
                # Train model
                step += 1
                for prompt in prompts:
                    step += 1                    
                    if prompt == "point":
                        batch_input = [
                            {'image': prep_img(image, resize_transform),
                             'point_coords':resize_transform.apply_coords_torch(torch.as_tensor(np.array([gen_points(mask[0,:])]), device=device), original_size=image.shape[:2]),
                             'point_labels':torch.as_tensor([[1]], device=device),
                             'original_size':image.shape[:2]
                             } 
                            for image, mask in batch_data
                        ]
                    if prompt == "bbox":
                        batch_input = [
                            {'image': prep_img(image, resize_transform),
                             'boxes':resize_transform.apply_boxes_torch(torch.as_tensor(np.array([gen_bboxes(mask[0,:])]), device=device), original_size=image.shape[:2]),
                             'original_size':image.shape[:2]
                             } 
                            for image, mask in batch_data
                        ]
                    batch_gt_masks = torch.as_tensor(np.array([mask for _, mask in batch_data]), dtype=torch.float, device=device)

                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        y_pred = samri_model(batch_input, multimask_output=False, train_mode=True)

                        focal_loss = sigmoid_focal_loss(y_pred, batch_gt_masks, alpha=0.25, gamma=2,reduction="mean")
                        loss = dice_loss(y_pred, batch_gt_masks) + 20 * focal_loss
                        loss /= 21

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    epoch_loss += loss.item()

                    experiment.log({"sub_loss": loss.item()})
                batch_data = []

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
            torch.save(samri_model.state_dict(), join(model_save_path, "samri_vitb_best_l40.pth"))

        ## save the latest model
        torch.save(samri_model.state_dict(), join(model_save_path, "samri_vitb_latest_l40.pth"))


if __name__ == "__main__":
    main()
           