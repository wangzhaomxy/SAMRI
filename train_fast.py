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
from utils.losses import DiceFocalLoss
from utils.utils import *
from utils.prompt import *
from model import SAMRI
from train_predictor import TrainSamPredictor
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


# setup global parameters
model_type = "samri"
encoder_type = ENCODER_TYPE[model_type] # choose one from vit_b and vit_h.
batch_size = BATCH_SIZE
model_save_path = MODEL_SAVE_PATH + "mult_sched/"
num_epochs = NUM_EPOCHS
train_image_path = TRAIN_IMAGE_PATH
train_image_path.remove('/scratch/project/samri/Embedding/totalseg_mr/')
save_every = 1

def gen_batch(mask, prompt):
    masks = MaskSplit(mask)
    lenth = len(masks)
    for each_mask in masks:
        if prompt == "point":
            each_prompt = gen_points(each_mask)
        if prompt == "bbox":
            each_prompt = gen_bboxes(each_mask)
        yield (each_mask, each_prompt, lenth)

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
    train_predictor = TrainSamPredictor(samri_model)

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
        lr=1e-4/world_size, 
        weight_decay=0.1
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    dice_focal_loass = DiceFocalLoss(sigmoid=True, 
                                     squared_pred=True, 
                                     reduction="mean",
                                     lambda_dice=1,
                                     lambda_focal=10)
    #train
    losses = []
    train_dataset = EmbDataset(train_image_path)
    train_loader = DataLoader(train_dataset, shuffle=False, sampler=DistributedSampler(train_dataset))

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

                        sub_mask = torch.tensor(sub_mask[None,:,:], dtype=torch.float, device=gpu)
                        loss = dice_focal_loass(y_pred, sub_mask)
                        loss.backward()
                        
                        optimizer.step()
                        
                        sub_loss += loss.item()
            epoch_loss += sub_loss / (len(prompts)*lenth)
        scheduler.step()
        epoch_loss /= (step+1)
        losses.append(epoch_loss)
        
        ## save the latest model
        if (epoch + 1) % save_every == 0 and gpu == 0:
            print(f"The {epoch+1} / {num_epochs} epochs,  Loss: {epoch_loss}.")
            torch.save(samri_model.module.state_dict(), join(model_save_path, f"samri_vitb_mult_{str(epoch+1)}.pth"))
            print(f"Checkpoint <samri_vitb_mult_{str(epoch+1)}.pth> has been saved.")
    destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, num_epochs, save_every), nprocs=world_size)
    main()