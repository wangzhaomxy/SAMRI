import numpy as np
import os
join = os.path.join
import torch
from segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F
from utils.utils import *
from tqdm import tqdm
from torch.utils.data import Dataset
import nibabel as nib
import glob, random

# From MedSAM Inference file
#################
@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().detach().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5) .astype(np.uint8)
    return medsam_seg
##################
# Code above are from MedSAM Inference file


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class NiiDataset(Dataset):
    def __init__(self, 
                 data_root, 
                 multi_mask=False):
        """
        Args:
            data_root (str): The path of the dataset
            multi_mask (bool): if Ture, return multi-labeld masks; if false, 
                                return a random mask from masks.
        """
        super().__init__()
        self.data_root = data_root
        self.img_file = []
        self.gt_file = []
        for path in self.data_root:
            self.img_file += sorted(glob.glob(path + IMAGE_KEYS))
            self.gt_file += sorted(glob.glob(path + MASK_KEYS))
        self.cur_name = ""
        self.cur_gt_name = ""
        self.multi_mask = multi_mask

    def __len__(self):
        return len(self.img_file)

    def __getitem__(self, index):
        # load input image and corresponding mask
        nii_img = self._load_nii(self.img_file[index])
        nii_seg = self._load_nii(self.gt_file[index])
        self.cur_name = self.img_file[index]
        self.cur_gt_name = self.gt_file[index]
        nii_img = nii_img[0, :, :]
        nii_seg = nii_seg[0, :, :]
        # shape of nii_img is (H,W), nii_seg is (H,W)
        if self.multi_mask:
            return (nii_img, nii_seg)
        else:
            return (nii_img, nii_seg==np.unique(nii_seg)[random.choice(np.unique(nii_seg).nonzero()[0])])

    def _load_nii(self, nii_file):
        """
        load nifty image, (C, H, W) = (1, 256, 256)

        parameters:
        nii_file(str): The input nifti file path.

        returns:
        (np.ndarray): The numpy format image, (C, H, W) = (1, 256, 256)
        """
        return nib.load(nii_file).get_fdata()
    
    def get_name(self):
        return os.path.basename(self.cur_name)

class MaskSplit():
    """
    Split the labeled ground truth masks into single binary mask. 

    Args:
        mask (np.darray): the labeled ground truth mask. CHW=(1,255,255)
        
    Returns:
        masks (list): A list of splited masks. HW=(255,255)
        
        labels (list): The list of splited masks labels.
    
    """
    def __init__(self, mask):
        self.mask = mask
        # mask_number (int): the number of the gt mask labels.
        self.mask_number = len(np.unique(self.mask)) - 1
        # masks (list): the list of single mask with different lables, HW=(255,255)
        self.masks, self.labels = self._split_masks()
        """
        Args:
            mask (np.darray): the labeled ground truth mask. CHW=(1,255,255)
            mask_number (int): the number of the gt mask labels.
            masks (list): the list of single mask with different lables, HW=(255,255)
        """

    def __len__(self):
        """
        The number of labels.
        """
        return self.mask_number
    
    def __getitem__(self, index):
        return self.masks[index], self.labels[index]
    
    def _split_masks(self):
        masks = []
        labels = []
        for label in np.unique(self.mask).nonzero()[0]:
            masks.append(self.mask == np.unique(self.mask)[label])
            labels.append(np.unique(self.mask)[label])
        return masks, labels
    
# Setup
file_paths = TEST_IMAGE_PATH
model_name = "MedSAM"
save_path = "/scratch/project/samri/MedSAM_inference/"

# From MedSAM Inference file
#################
device = "cuda"
ckpt = "/scratch/user/s4670484/Model_dir/medsam_vit_b.pth"
medsam_model = sam_model_registry["vit_b"](checkpoint=ckpt)
medsam_model = medsam_model.to(device)
medsam_model.eval()
##################
# Code above are from MedSAM Inference file

for file_path in file_paths:
    print("Processing the dataset: ",file_path)
    ds_name = file_path.split("/")[-3]
    test_dataset = NiiDataset([file_path], multi_mask= True)
    for img_np, masks in tqdm(test_dataset):
        try:
            img_name = test_dataset.get_name()
            
            # Preprocessing image, from MedSAM Inference file
            ###################
            # img_np = io.imread(args.data_path)
            if len(img_np.shape) == 2:
                img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
            else:
                img_3c = img_np
            H, W, _ = img_3c.shape
            img_1024 = transform.resize(
                img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
            ).astype(np.uint8)
            img_1024 = (img_1024 - img_1024.min()) / np.clip(
                img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
            )  # normalize to [0, 1], (H, W, 3)
            # convert the shape to (3, H, W)
            img_1024_tensor = (
                torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
            )
            #####################
            # Code above are from MedSAM Inference file
            
            # Generate bounding box for every mask
            for mask, label in MaskSplit(masks):
                gt_seg = mask
                
                # From MedSAM train_multi_gpus.py file
                ###################
                # Generate box prompt from gt mask
                gt2D = np.uint8(mask)# only one label, (256, 256)

                assert np.max(gt2D) == 1 and np.min(gt2D) == 0.0, "ground truth should be 0, 1"
                y_indices, x_indices = np.where(gt2D > 0)
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                # add perturbation to bounding box coordinates
                H, W = gt2D.shape
                # No jitter for testing, comment off the following parts
                # x_min = max(0, x_min - random.randint(0, self.bbox_shift))
                # x_max = min(W, x_max + random.randint(0, self.bbox_shift))
                # y_min = max(0, y_min - random.randint(0, self.bbox_shift))
                # y_max = min(H, y_max + random.randint(0, self.bbox_shift))
                bboxes = np.array([x_min, y_min, x_max, y_max])
                #####################
                # Code above are from MedSAM train_multi_gpus.py file
                
                # From MedSAM Inference file
                ###################
                # transfer box_np t0 1024x1024 scale
                box_1024 = [bboxes] / np.array([W, H, W, H]) * 1024               
                
                with torch.no_grad():
                    image_embedding = medsam_model.image_encoder(img_1024_tensor)  # (1, 256, 64, 64)

                medsam_seg = medsam_inference(medsam_model, image_embedding, box_1024, H, W)
                #####################
                # Code above are from MedSAM Inference file

                comb_seg = np.concatenate([gt_seg, medsam_seg], axis=1) *255 # For visualize observations

                # Create new folders to save results
                ds_dir = save_path + ds_name + "/"
                make_dir(ds_dir)
                result_dir = ds_dir + "results/"
                make_dir(result_dir)
                comb_dir = ds_dir + "comb/"
                make_dir(comb_dir)
                
                # From MedSAM Inference file
                ###################
                io.imsave(
                    join(comb_dir, "comb_" + img_name[:-7] + ".png"),
                    comb_seg,
                    check_contrast=False,
                )
                #####################
                # Code above are from MedSAM Inference file
                
                # Save the ground truth and inference results in .npz files.
                np.savez_compressed(result_dir + img_name[:-7] + ".npz", 
                                    gt=gt_seg, 
                                    medsam=medsam_seg)
        except Exception as e:
            print(e)
            print("Error in file: " + test_dataset.cur_name)
            continue