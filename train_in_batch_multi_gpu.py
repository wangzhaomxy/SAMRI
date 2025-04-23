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
from utils.dataloader import EmbDataset, BalancedEmbDataset
from torch.utils.data import DataLoader
from utils.losses import DiceFocalLoss
from utils.utils import *
from utils.prompt import *
from model import SAMRI
from segment_anything.utils.transforms import ResizeLongestSide
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# setup global parameters
model_type = "samri"
encoder_type = ENCODER_TYPE[model_type] # choose one from vit_b and vit_h.
batch_size = BATCH_SIZE
model_save_path = MODEL_SAVE_PATH + "box-501_balance/"
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
num_epochs = NUM_EPOCHS
train_image_path = TRAIN_IMAGE_PATH
train_image_path.remove('/scratch/project/samri/Embedding/totalseg_mr/')
save_every = 1

def ddp_setup(rank: int, world_size: int):
    """
    Args:
    rank: Unique identifier of each process
    world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def main(gpu, world_size, num_epochs, save_every):
    ddp_setup(rank=gpu, world_size=world_size)
    sam_checkpoint, start_epoch = get_checkpoint(model_save_path)
    sam_model = sam_model_registry[encoder_type](sam_checkpoint)
    samri_model = SAMRI(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).cuda()
    resize_transform = ResizeLongestSide(samri_model.image_encoder.img_size)
    
    if gpu == 0:
        print(
            "Number of total parameters: ",
            sum(p.numel() for p in samri_model.parameters()),
        )  
        print(
            "Number of trainable parameters: ",
            sum(p.numel() for p in samri_model.parameters() if p.requires_grad),
        )

    samri_model = DDP(
                    samri_model,
                    device_ids=[gpu],
                    gradient_as_bucket_view=True,
                    find_unused_parameters=True
                    )

    optimizer = torch.optim.AdamW(
        samri_model.module.mask_decoder.parameters(),
        lr=1e-5,
        weight_decay=0.1
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    dice_focal_loss = DiceFocalLoss(sigmoid=True, 
                                     squared_pred=True,
                                     batch= True, 
                                     reduction="mean",
                                     lambda_dice=1,
                                     lambda_focal=20)
    
    #train
    losses = []
    # train_dataset = EmbDataset(train_image_path, 
    #                            resize_mask=True, 
    #                            mask_size=256)
    train_image_path = "/scratch/project/samri/train_list.pkl"
    train_dataset = BalancedEmbDataset(train_image_path, 
                               sub_set="60_up",
                               resize_mask=True, 
                               mask_size=256)
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=False, 
                              sampler=DistributedSampler(train_dataset))
    
    prompts = ["bbox"] #  ["point", "bbox"]
    for epoch in range(start_epoch, num_epochs):
        # training part
        samri_model.train()
        epoch_loss = 0
        for prompt in prompts:
            for step, (embedding, masks, ori_size) in enumerate(tqdm(train_loader)):
                # Train model
                ori_size = [(ori_size[0].numpy()[i], ori_size[1].numpy()[i]) for i in range(len(ori_size[0]))]
                
                step += 1
                if prompt == "point":
                    batch_input = [
                        {'image': image.squeeze(),
                            'point_coords':resize_transform.apply_coords_torch(torch.as_tensor(np.array([gen_points(mask.squeeze(0).numpy())]), device=gpu), original_size=ori_size),
                            'point_labels':torch.as_tensor([[1]]),
                            'original_size':ori_size
                            } 
                        for image, mask, ori_size in zip(embedding, masks, ori_size)
                    ]
                if prompt == "bbox":
                    batch_input = [
                        {'image': image.squeeze(),
                            'boxes':resize_transform.apply_boxes_torch(torch.as_tensor(np.array([gen_bboxes(mask.squeeze(0).numpy(),jitter=JITTER)]), device=gpu), original_size=ori_size),
                            'original_size':ori_size
                            } 
                        for image, mask, ori_size in zip(embedding, masks, ori_size)
                    ]

                y_pred = samri_model(batch_input, multimask_output=False, train_mode=True, embedding_inputs=True)
                loss = dice_focal_loss(y_pred, masks.to(gpu))
                loss.backward()
                optimizer.step()

                optimizer.zero_grad()
                epoch_loss += loss.item()
        # scheduler.step()
        epoch_loss /= step
        losses.append(epoch_loss)

        ## save the latest model
        if (epoch + 1) % save_every == 0 and gpu == 0:
            print(f"The {epoch+1} / {num_epochs} epochs,  Loss: {epoch_loss/21:.4f}.")
            torch.save(samri_model.module.state_dict(), join(model_save_path, f"samri_vitb_box_{str(epoch+1)}.pth"))
            print(f"Checkpoint <samri_vitb_box_{str(epoch+1)}.pth> has been saved.")
    destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, num_epochs, save_every), nprocs=world_size)
        