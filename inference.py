from segment_anything import sam_model_registry
from segment_anything import SamPredictor
import os
import numpy as np


model_type = 'vit_b'# Choose one from vit_b, vit_h, samri, and med_sam
encoder_tpye = ENCODER_TYPE[model_type] 
checkpoint = ckpt
device = DEVICE
print("Testing Check-point: " + ckpt)

# regist the MRI-SAM model and predictor.
sam_model = sam_model_registry[encoder_tpye](checkpoint)
sam_model = sam_model.to(device)
sam_model.eval()

save_infer_results(file_paths=file_paths,
                sam_model=sam_model, 
                save_path=save_path1 + model_name + "/")



def save_infer_outputs_from_ds(model, test_dataset, save_path, ds_name):
    """
    Save SAM model inference and ground truth results as image/npz files.
    Args:
        model: SAM model.
        test_dataset: PyTorch Dataset loader.
        save_path: Base path to save outputs.
        ds_name: Dataset name for folder structuring.
    """
    predictor = SamPredictor(model)

    for image, mask, img_fullpath, _ in tqdm(test_dataset):
        image = image.squeeze(0).detach().cpu().numpy()
        mask = mask.squeeze(0).detach().cpu().numpy()
        H, W = mask.shape[-2:]

        predictor.set_image(image)
        
        if isinstance(img_fullpath, (tuple, list)):
            img_fullpath = img_fullpath[0]
        img_name = os.path.basename(img_fullpath).replace(".nii.gz", "")

        for each_mask, label in MaskSplit(mask):
            gt_seg = each_mask
            bbox = gen_bboxes(each_mask, jitter=0)
            point = gen_points(each_mask)
            point_label = np.array([1])

            pre_mask_b, _, _ = predictor.predict(
                point_coords=point,
                point_labels=point_label,
                box=bbox[None, :],
                multimask_output=False,
            )

            pre_seg = pre_mask_b[0, :, :]
            comb_seg = np.concatenate([gt_seg, pre_seg], axis=1) * 255
            
            # Save results
            ds_dir = os.path.join(save_path, ds_name)
            result_dir = os.path.join(ds_dir, "results")
            comb_dir = os.path.join(ds_dir, "comb")
            make_dir(result_dir)
            make_dir(comb_dir)

            io.imsave(
                os.path.join(comb_dir, f"comb_{img_name}_{label}.png"),
                comb_seg.astype(np.uint8),
                check_contrast=False,
            )

            np.savez_compressed(
                os.path.join(result_dir, f"{img_name}_{label}.npz"),
                img = image[..., 0],
                gt=gt_seg,
                pred=pre_seg,
            )