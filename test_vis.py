from segment_anything import sam_model_registry
from utils.visual import get_dice_from_ds
from utils.utils import *
from utils.dataloader import NiiDataset
import pickle
from tqdm import tqdm


file_paths = TEST_IMAGE_PATH
ckpt_root_path = "/scratch/project/samri/Model_save/"
model_folder = "bp11/"
ckpt_list = [ckpt_root_path + model_folder + "samri_vitb_bp11_0.pth",
             ckpt_root_path + model_folder + "samri_vitb_bp11_1.pth",
             ckpt_root_path + model_folder + "samri_vitb_bp11_2.pth",
             ckpt_root_path + model_folder + "samri_vitb_bp11_3.pth",
             ckpt_root_path + model_folder + "samri_vitb_bp11_4.pth",
             ckpt_root_path + model_folder + "samri_vitb_bp11_5.pth",
             ckpt_root_path + model_folder + "samri_vitb_bp11_6.pth",
             ckpt_root_path + model_folder + "samri_vitb_bp11_7.pth",
             ]

def save_test_record(file_paths, sam_model, save_path):
    p_record = []
    b_record = []
    for file_path in file_paths:
        print("Processing the dataset: ",file_path)
        test_dataset = NiiDataset([file_path], multi_mask= True)    
        p_record_vitb, b_record_vitb = get_dice_from_ds(model=sam_model, test_dataset=test_dataset)
        p_record.append(p_record_vitb)
        b_record.append(b_record_vitb)
        final_record = {"p":p_record,"b":b_record}
    with open(save_path, "wb") as f:
        pickle.dump(final_record, f)

for ckpt in ckpt_list:
    model_type = 'vit_b'# Choose one from vit_b, vit_h, samri, and med_sam
    encoder_tpye = ENCODER_TYPE[model_type] 
    checkpoint = ckpt
    device = DEVICE
    file_name = ckpt.split("/")[-1]
    print("Testing Check-point " + file_name)

    # regist the MRI-SAM model and predictor.
    sam_model = sam_model_registry[encoder_tpye](checkpoint)
    sam_model = sam_model.to(device)
    save_path = "/scratch/project/samri/Eval_results/" + model_folder + file_name[:-4]

    save_test_record(file_paths, sam_model, save_path)