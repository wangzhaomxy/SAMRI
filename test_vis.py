from segment_anything import sam_model_registry
from utils.visual import save_test_record, save_test_record_from_emb
from utils.utils import *


file_paths = TEST_EMB_PATH
ckpt_root_path = "/scratch/project/samri/Model_save/"
model_folder = "box/"
# model_folder = "box-501_balance/"
save_path = "/scratch/project/samri/Eval_results/" + model_folder
# ckpt_list = [
            #  ckpt_root_path + model_folder + "samri_vitb_box_0.pth",
            #  ckpt_root_path + model_folder + "samri_vitb_box_1.pth",
            #  ckpt_root_path + model_folder + "samri_vitb_box_2.pth",
            #  ckpt_root_path + model_folder + "samri_vitb_box_3.pth",
            #  ckpt_root_path + model_folder + "samri_vitb_box_4.pth",
            #  ckpt_root_path + model_folder + "samri_vitb_box_5.pth",
            #  ckpt_root_path + model_folder + "samri_vitb_box_6.pth",
            #  ckpt_root_path + model_folder + "samri_vitb_box_7.pth",
            #  ckpt_root_path + model_folder + "samri_vitb_box_8.pth",
            #  ckpt_root_path + model_folder + "samri_vitb_box_9.pth",
            #  ckpt_root_path + model_folder + "samri_vitb_box_10.pth",
            #  ckpt_root_path + model_folder + "samri_vitb_box_11.pth",
            #  ckpt_root_path + model_folder + "samri_vitb_box_23.pth",
            #  ckpt_root_path + model_folder + "samri_vitb_box_24.pth",
            #  ckpt_root_path + model_folder + "samri_vitb_box_19.pth",
            #  ckpt_root_path + model_folder + "samri_vitb_box_24.pth",
            #  ]
ckpt_list = ["/scratch/user/s4670484/Model_dir/sam_vit_b_01ec64.pth"]
# ckpt_list = ["/scratch/user/s4670484/Model_dir/sam_vit_h_4b8939.pth"]
# ckpt_list = ["/scratch/user/s4670484/Model_dir/samri_vitb.pth"]

for ckpt in ckpt_list:
    model_type = 'vit_b'# Choose one from vit_b, vit_h, samri, and med_sam
    encoder_tpye = ENCODER_TYPE[model_type] 
    checkpoint = ckpt
    device = DEVICE
    model_name = ckpt.split("/")[-1]
    print("Testing Check-point " + model_name)

    # regist the MRI-SAM model and predictor.
    sam_model = sam_model_registry[encoder_tpye](checkpoint)
    sam_model = sam_model.to(device)
    sam_model.eval()
    save_path_all = save_path + model_name[:-4]

    save_test_record_from_emb(file_paths=file_paths,
                     sam_model=sam_model, 
                     save_path=save_path_all, 
                     by_ds=False)


# save_pxl_record(file_paths, save_path)
# print("Done!")