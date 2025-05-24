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
import pandas as pd
from segment_anything import sam_model_registry
from utils.dataloader import EmbDataset
from torch.utils.data import DataLoader
from utils.losses import DiceLoss
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
model_path = MODEL_SAVE_PATH + "sam_vitb/"
val_emb_path = VAL_EMBEDDING_PATH
model_files = [f for f in os.listdir(model_path) if f.startswith("samri_vitb_box_")]

def get_epoch_num(filename):
    """
    Extract the epoch number from the filename.
    Example: 'samri_vitb_box_10.pth' -> 10
    """
    match = filename.split('_')[-1].split('.')[0]
    if match:
        return int(match)
    return None

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

def main(gpu, world_size):
    ddp_setup(rank=gpu, world_size=world_size)

    if gpu == 0:
        print("Number of GPUs: ", world_size)
        print("Batch size: ", batch_size)
        
    val_dataset = EmbDataset(val_emb_path, 
                               resize_mask=True, 
                               mask_size=256)
    num_workers = 8
    val_loader = DataLoader(val_dataset, 
                              batch_size=batch_size, 
                              shuffle=False,
                              num_workers=num_workers,
                              sampler=DistributedSampler(val_dataset))
 
    dice_loss = DiceLoss(sigmoid=True, 
                                     squared_pred=True,
                                     batch= True, 
                                     reduction="mean")    

    # Validate the model files
    losses = []
    for model_file in model_files:
        epoch = get_epoch_num(model_file)
        print(f"Validating model from epoch {epoch}...")

        # Load model
        sam_model = sam_model_registry[encoder_type](model_path + model_file)
        samri_model = SAMRI(
            image_encoder=sam_model.image_encoder,
            mask_decoder=sam_model.mask_decoder,
            prompt_encoder=sam_model.prompt_encoder,
        ).to(gpu)
        resize_transform = ResizeLongestSide(samri_model.image_encoder.img_size)   
        samri_model = DDP(
                        samri_model,
                        device_ids=[gpu],
                        gradient_as_bucket_view=True,
                        find_unused_parameters=True
                        )
        samri_model.eval()
        loss_list = []
        with torch.no_grad():
            for embeddings, masks, ori_size in tqdm(val_loader, desc=f"Epoch {epoch}"):
                ori_size = [(ori_size[0][i].item(), ori_size[1][i].item()) for i in range(len(ori_size[0]))]
                batch_input = [
                    {
                        'image': emb.squeeze(),
                        'boxes': resize_transform.apply_boxes_torch(
                            torch.as_tensor(np.array(gen_bboxes(mask.squeeze(0).numpy(), 
                                                        jitter=0), device=gpu)),
                                                        original_size=(256, 256)
                        ),
                        'original_size': size,
                    }
                    for emb, mask, size in zip(embeddings, masks, ori_size)
                ]

                preds = samri_model(batch_input, multimask_output=False, train_mode=False, embedding_inputs=True)

                loss = dice_loss(preds.cpu().numpy(), masks.cpu().numpy())
                loss_list.append(loss)
        avg_loss = np.mean(loss_list)
        print(f"Epoch {epoch}: Mean Loss = {avg_loss:.4f}")
        losses.append((epoch, avg_loss))
    
    # Save results
    if gpu == 0:
        
        df = pd.DataFrame(losses, columns=["Epoch", "DiceScore"])
        df.set_index("Epoch", inplace=True)
        df.sort_index(inplace=True)
        
        result_path = join(model_path, "validation_results")
        os.makedirs(result_path, exist_ok=True)
        df.to_csv(result_path)
        print(f"Validation results saved to {result_path}")
        
    destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size)
        