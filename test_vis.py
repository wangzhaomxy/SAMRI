from segment_anything import sam_model_registry
from utils.visual import *
from utils.utils import *
import time

# file_paths = TEST_IMAGE_PATH + TEST_ZEROSHOT_PATH
file_paths = TEST_IMAGE_PATH + TEST_ZEROSHOT_PATH
ckpt_root_path = "/scratch/project/samri/Model_save/"
model_folder = "bp_fullds_balance_up/"
# model_folder = "fullds_balance_up_new_loss/"
# save_path = "/scratch/project/samri/Eval_results/" + model_folder
save_path = "/scratch/project/samri/Eval_results/" + "bp_fullds_balance_up_zero/"
make_dir(save_path)

# save_path1 = "/scratch/project/samri/Inference_results/" + "SAMRI/"
# make_dir(save_path1)

ckpt_list = [
             ckpt_root_path + model_folder + "samri_vitb_box_1.pth",
             ckpt_root_path + model_folder + "samri_vitb_box_2.pth",
             ckpt_root_path + model_folder + "samri_vitb_box_3.pth",
             ckpt_root_path + model_folder + "samri_vitb_box_4.pth",
             ckpt_root_path + model_folder + "samri_vitb_box_5.pth",
             ckpt_root_path + model_folder + "samri_vitb_box_6.pth",
             ckpt_root_path + model_folder + "samri_vitb_box_7.pth",
             ckpt_root_path + model_folder + "samri_vitb_box_8.pth",
             ckpt_root_path + model_folder + "samri_vitb_box_9.pth",
             ckpt_root_path + model_folder + "samri_vitb_box_10.pth",
             ckpt_root_path + model_folder + "samri_vitb_box_20.pth",
             ckpt_root_path + model_folder + "samri_vitb_box_30.pth",
             ckpt_root_path + model_folder + "samri_vitb_box_40.pth",
             ckpt_root_path + model_folder + "samri_vitb_box_50.pth",
             ]
# ckpt_list = ["/scratch/user/s4670484/Model_dir/sam_vit_b_01ec64.pth"]
# ckpt_list = ["/scratch/user/s4670484/Model_dir/sam_vit_h_4b8939.pth"]
# ckpt_list = ["/scratch/user/s4670484/Model_dir/samri_vitb.pth"]

for ckpt in ckpt_list:
    start = time.time()
    model_type = 'vit_b'# Choose one from vit_b, vit_h, samri, and med_sam
    encoder_tpye = ENCODER_TYPE[model_type] 
    checkpoint = ckpt
    device = DEVICE
    model_name = ckpt.split("/")[-1]
    print("Testing Check-point: " + ckpt)

    # regist the MRI-SAM model and predictor.
    sam_model = sam_model_registry[encoder_tpye](checkpoint)
    sam_model = sam_model.to(device)
    sam_model.eval()
    save_path_all = save_path + model_name[:-4]

    save_test_record(file_paths=file_paths,
                     sam_model=sam_model, 
                     save_path=save_path_all)
    
    # save_infer_results(file_paths=file_paths,
    #                    sam_model=sam_model, 
    #                    save_path=save_path1)
    end = time.time()
    print(f"Elapsed time: {end - start:.2f} seconds")
    print("Done!")