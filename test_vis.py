from segment_anything import sam_model_registry
from utils.visual import get_dice_from_ds, get_pix_num_from_ds
from utils.utils import *
from utils.dataloader import NiiDataset
import pickle
from tqdm import tqdm


file_paths = TEST_IMAGE_PATH
ckpt_root_path = "/scratch/project/samri/Model_save/"
model_folder = "base/"
save_path = "/scratch/project/samri/Eval_results/" + model_folder
# ckpt_list = [ckpt_root_path + model_folder + "samri_vitb_box_0.pth",
#              ckpt_root_path + model_folder + "samri_vitb_box_1.pth",
#              ckpt_root_path + model_folder + "samri_vitb_box_2.pth",
#              ckpt_root_path + model_folder + "samri_vitb_box_3.pth",
#              ckpt_root_path + model_folder + "samri_vitb_box_4.pth",
#              ckpt_root_path + model_folder + "samri_vitb_box_5.pth",
#              ckpt_root_path + model_folder + "samri_vitb_box_6.pth",
#              ckpt_root_path + model_folder + "samri_vitb_box_7.pth",
#              ckpt_root_path + model_folder + "samri_vitb_box_8.pth",
#              ckpt_root_path + model_folder + "samri_vitb_box_9.pth",
#              ckpt_root_path + model_folder + "samri_vitb_box_10.pth",
#              ]
# ckpt_list = ["/scratch/user/s4670484/Model_dir/sam_vit_b_01ec64.pth"]
ckpt_list = ["/scratch/user/s4670484/Model_dir/sam_vit_h_4b8939.pth"]
# ckpt_list = ["/scratch/user/s4670484/Model_dir/medsam_vit_b.pth"]

def save_test_record(file_paths, sam_model, save_path):
    p_record, b_record = [], []
    pixel_count, area_percentage = [], []
    for file_path in file_paths:
        print("Processing the dataset: ",file_path)
        test_dataset = NiiDataset([file_path], multi_mask= True)    
        (p_record_vitb, 
         b_record_vitb, 
         pixel_count_vit, 
         area_percentage_vit) = get_dice_from_ds(model=sam_model, 
                                                 test_dataset=test_dataset, 
                                                 med_sam=False,
                                                 with_pix=True)
        p_record.append(p_record_vitb)
        b_record.append(b_record_vitb)
        pixel_count.append(pixel_count_vit)
        area_percentage.append(area_percentage_vit)
        final_record = {"p":p_record, 
                        "b":b_record, 
                        "pixel_count":pixel_count,
                        "area_percentage":area_percentage}
    with open(save_path, "wb") as f:
        pickle.dump(final_record, f)

for ckpt in ckpt_list:
    model_type = 'vit_b'# Choose one from vit_b, vit_h, samri, and med_sam
    encoder_tpye = ENCODER_TYPE[model_type] 
    checkbox = ckpt
    device = DEVICE
    file_name = ckpt.split("/")[-1]
    print("Testing Check-point " + file_name)

    # regist the MRI-SAM model and predictor.
    sam_model = sam_model_registry[encoder_tpye](checkbox)
    sam_model = sam_model.to(device)
    save_path_all = save_path + file_name[:-4]

    save_test_record(file_paths, sam_model, save_path_all)


# def save_pxl_record(file_paths, save_path):
#     pixel_count, area_percentage = [], []
#     for file_path in file_paths:
#         print("Processing the dataset: ",file_path)
#         test_dataset = NiiDataset([file_path], multi_mask= True)
#         pixel_count_vit, area_percentage_vit = get_pix_num_from_ds(test_dataset=test_dataset)
#         pixel_count.append(pixel_count_vit)
#         area_percentage.append(area_percentage_vit)
#         final_record = {"pixel_count":pixel_count,"area_percentage":area_percentage}
#     with open(save_path, "wb") as f:
#         pickle.dump(final_record, f)

# save_pxl_record(file_paths, save_path)
# print("Done!")