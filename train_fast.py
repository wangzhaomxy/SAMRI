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
from utils.dataloader import emb_name_split
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
sam_checkpoint = sorted(glob.glob(MODEL_SAVE_PATH + "*"))[-1]
batch_size = BATCH_SIZE
data_path = TRAIN_IMAGE_PATH
model_save_path = MODEL_SAVE_PATH
device = DEVICE
num_epochs = NUM_EPOCHS
train_image_path = TRAIN_IMAGE_PATH


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

    # train
    losses = []
    train_files = emb_name_split(train_image_path)
    rounds = num_epochs // NUM_EPO_PER_ROUND
    start_epoch = int(os.path.basename(sam_checkpoint)[:-4].split('_')[-1])
    prompts = ["point", "bbox"]
    
    for rou in range(rounds):
        print(f"The {rou+1} / {rounds} rounds.")
        
        for sub_set in train_files:
            train_dataset = []
            for name in sub_set:
                file = np.load(name)
                train_dataset.append(file)
                file.close()
            
            for epoch in range(NUM_EPO_PER_ROUND):
                # training part
                samri_model.train()
                epoch_loss = 0
                for step, (embedding, mask, ori_size) in enumerate(tqdm(train_dataset)):
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
                                loss = dice_loss(y_pred, sub_mask) + focal_loss
                                
                                loss.backward()
                                
                                optimizer.step()
                                
                                sub_loss += loss.item()
                    epoch_loss += sub_loss / (len(prompts)*lenth)

                epoch_loss /= step
                losses.append(epoch_loss)


        print(
            f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}'
        )
        torch.save(samri_model.state_dict(), join(model_save_path, "samri_vitb_fast", start_epoch+(rou+1)*NUM_EPO_PER_ROUND, ".pth"))
            


if __name__ == "__main__":
    main()