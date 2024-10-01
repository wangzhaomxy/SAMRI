

from tqdm import tqdm
import processing_utile as pu
import glob


def process_BTDF(input_path, output_path):
    data_name = pu.sort_all_fnames(input_path + "/data/")
    
    folder_name = output_path + "/" + pu.fname_from_path(input_path)
    save_path_out, save_img_path_out = pu.create_save_folder(folder_name)
    
    for data in tqdm(data_name):
        filename, image, label, mask = pu.read_BTDF_file(data)
        pu.save_nib_data(save_path_out, filename, image, mask, label)
        pu.save_img_data(save_img_path_out, filename, image, mask, label)

def process_MSD(input_path, output_path):
    sub_root_paths = pu.sort_all_fnames(input_path)
    for sub_root_path in sub_root_paths:
        ds_info = pu.read_json(sub_root_path)
        data_set = ds_info["training"]

        folder_name = output_path + "/MSD/" + \
                                            pu.fname_from_path(sub_root_path)
        save_path_out, save_img_path_out = pu.create_save_folder(folder_name)

        pu.process_nii_json(data_set, sub_root_path, ds_info,
                            save_path_out, save_img_path_out)

def process_amos22(input_path, output_path):
    input_path = input_path + "/amos22"
    data_folder=["training", "validation"]
    ds_info = pu.read_json(input_path)
    
    folder_name = output_path  + "/" +pu.fname_from_path(input_path)
    save_path_out, save_img_path_out = pu.create_save_folder(folder_name)
    
    for dfolder in data_folder:
        data_set = ds_info[dfolder]
        pu.process_nii_json(data_set, input_path, ds_info,
                            save_path_out, save_img_path_out)
        
def process_MSLite(input_path, output_path):
    sub_root_paths = pu.sort_all_fnames(input_path)
    for sub_root_path in sub_root_paths:
        data_name = pu.sort_all_fnames(sub_root_path)
        
        folder_name = output_path + "/MedSamLite/" + \
                                            pu.fname_from_path(sub_root_path)
        save_path_out, save_img_path_out = pu.create_save_folder(folder_name)
        
        for data in tqdm(data_name):
            filename, image, mask = pu.read_npz(data)
            pu.slice_np(image=image, mask=mask, filename=filename,
                         threshold=10,
                         save_nii_path_out=save_path_out,
                         save_img_path_out=save_img_path_out)

def process_MSK(input_path, output_path):
    sub_root_paths = pu.sort_all_fnames(input_path)
    for sub_root_path in sub_root_paths:
        img_name = sorted(glob.glob(sub_root_path + "/*img*.gz"))
        mask_name = sorted(glob.glob(sub_root_path + "/*seg*.gz"))
        
        folder_name = output_path + "/MSK_Knee/" + \
                                            pu.fname_from_path(sub_root_path)
        save_path_out, save_img_path_out = pu.create_save_folder(folder_name)
        
        for img_path, mask_path in tqdm(list(zip(img_name,mask_name))):
            image = pu.read_nii_file(img_path)
            mask = pu.read_nii_file(mask_path)
            filename = pu.fname_from_path(img_path)[:-25]
            pu.slice_np(image=image, mask=mask, filename=filename,
                         threshold=50,
                         save_nii_path_out=save_path_out,
                         save_img_path_out=save_img_path_out)

def process_PROMISE(input_path, output_path):
    pass

def process_ACDC(input_path, output_path):
    pass

def process_Brain_TR_G(input_path, output_path):
    pass

def porcess_CHAOS(input_path, output_path):
    pass

def porcess_Meningioma(input_path, output_path):
    pass

def process_NCI_ISBI(input_path, output_path):
    pass

def process_Picai(input_path, output_path):
    pass

def process_QUBIQ(input_path, output_path):
    pass






