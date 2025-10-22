# -*- coding: utf-8 -*-
"""
train the mask decoder(multi-GPU)
freeze image encoder and prompt encoder

Usage example:
python train_multi_gpus.py \
  --model_type samri \
  --batch_size 48 \
  --data_path ./user_data \
  --model_save_path /user_data/Model_save \
  --num-epochs 120 \
  --save-every 2 \
  --prompts mixed \

"""

# --- Standard library
import os
import argparse
from os.path import join 

# --- Third-party
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from tqdm import tqdm
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

# --- Local project
from utils.dataloader import EmbDataset
from utils.losses import DiceFocalLoss
from utils.utils import *            
from model import SAMRI

# setup constant parameters
cfg = SAMRIConfig()

# setup global parameters and converted to CLI-driven.
_parser = argparse.ArgumentParser(add_help=True)
_parser.add_argument("--model_type", default="samri", choices=list(cfg.ENCODER_TYPE.keys()),
                     help="Model key used to derive encoder_type from ENCODER_TYPE.")
_parser.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
_parser.add_argument("--data_path", type=str, default=cfg.root_path, 
                     help="The root path of the dataset and user data.")
_parser.add_argument("--model_save_path", default=cfg.MODEL_SAVE_PATH, 
                     help="The path to save the trained model.") 
_parser.add_argument("--num_epochs", type=int, default=cfg.NUM_EPOCHS, 
                     help="Number of training epochs.")
_parser.add_argument("--save_every", type=int, default=1, 
                     help="Save the model every N epochs.")
_parser.add_argument("--prompts", nargs="+", default=["mixed"], choices=["point","bbox","mixed"],
                     help = "Prompt types for training, choose from 'point', 'bbox', and 'mixed'. 'mixed' means both point and bbox prompts.") 
_args, _unknown = _parser.parse_known_args()

cfg.root_path = _args.data_path
model_type = _args.model_type
encoder_type = cfg.ENCODER_TYPE[model_type] 
batch_size = _args.batch_size
model_save_path = _args.model_save_path
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path, exist_ok=True)
num_epochs = _args.num_epochs
save_every = _args.save_every
prompts = _args.prompts
train_embedding_paths = cfg.TRAIN_EMBEDDING_PATH

def ddp_setup(rank: int, world_size: int):
    """
    Args:
    rank: Unique identifier of each process
    world_size: Total number of processes
    """
    master_port = os.environ.get("MASTER_PORT", "29500")
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = master_port
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def main(gpu, world_size, num_epochs, save_every):
    ddp_setup(rank=gpu, world_size=world_size)
    sam_checkpoint, start_epoch = get_checkpoint(model_save_path, cfg.SAM_CHECKPOINT["vit_b"])
    sam_model = sam_model_registry[encoder_type](sam_checkpoint)
    samri_model = SAMRI(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(gpu)
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
        print("Number of GPUs: ", world_size)
        print("Batch size: ", batch_size)
        print("The model will be saved to: ", model_save_path)

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

    dice_focal_loss = DiceFocalLoss(sigmoid=True, 
                                     squared_pred=True,
                                     reduction="mean",
                                     lambda_dice=1,
                                     lambda_focal=10)
    
    #train
    losses = []
    train_dataset = EmbDataset(train_embedding_paths,
                               random_mask=True, 
                               resize_mask=True, 
                               mask_size=256)

    num_workers = 8
    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=False,
                              num_workers=num_workers,
                              sampler=DistributedSampler(train_dataset))
    
    for epoch in range(start_epoch, num_epochs):
        # training part
        samri_model.train()
        train_loader.sampler.set_epoch(epoch)
        epoch_loss = 0
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
                            'boxes':resize_transform.apply_boxes_torch(torch.as_tensor(np.array(gen_bboxes(mask.squeeze(0).numpy(),jitter=cfg.JITTER)), device=gpu), original_size=(256, 256)),
                            'original_size':ori_size
                            } 
                        for image, mask, ori_size in zip(embedding, masks, ori_size)
                    ]
                if prompt == "mixed":
                    batch_input = [
                        {'image': image.squeeze(),
                            'point_coords':resize_transform.apply_coords_torch(torch.as_tensor(np.array(gen_points(mask.squeeze(0).numpy())), device=gpu), original_size=(256, 256)),
                            'point_labels':torch.as_tensor([1]),
                            'boxes':resize_transform.apply_boxes_torch(torch.as_tensor(np.array(gen_bboxes(mask.squeeze(0).numpy(),jitter=cfg.JITTER)), device=gpu), original_size=(256, 256)),
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
        if (epoch + 1) % save_every == 0 and gpu == 0:
            print(f"The {epoch+1} / {num_epochs} epochs,  Loss: {epoch_loss:.4f}.")
            torch.save(samri_model.module.state_dict(), join(model_save_path, f"samri_vitb_bp_{str(epoch+1)}.pth"))
            print(f"Checkpoint <samri_vitb_bp_{str(epoch+1)}.pth> has been saved.")
    destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, num_epochs, save_every), nprocs=world_size)
        