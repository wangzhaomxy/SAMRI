

from tqdm import tqdm
import processing_utile as pu
from glob import glob
import random
import numpy as np


def process_BTDF(input_path, output_path):
    """Process the Brain Tumor Dataset Figshare data.

    Args:
        input_path (str): The data path. Example: "X:/xxxx/xxxx/xxxx"
        output_path (str): The saving path. Example: "X:/xxxx/xxxx/xxxx"
    """
    data_name = pu.sort_all_fnames(input_path + "/data/")  #data list
    
    # Create folders
    ds_name = pu.fname_from_path(input_path)
    save_path_out, save_img_path_out = pu.create_save_folder(output_path, ds_name)
    
    # Patientwise split training/validation/testing files
    patient_ids = []
    for data in tqdm(data_name):#reading pids
        _, _, _, _, pid = pu.read_BTDF_file(data)
        patient_ids.append(pid)
    patient_ids = list(set(patient_ids))
    random.shuffle(patient_ids)
    train_ids = patient_ids[:int(len(patient_ids)*0.8)]
    val_ids = patient_ids[int(len(patient_ids)*0.8):int(len(patient_ids)*0.9)]
    test_ids = patient_ids[int(len(patient_ids)*0.9):]
    
    # Save Data
    for data in tqdm(data_name):
        filename, image, label, mask, pid = pu.read_BTDF_file(data)
        mask = pu.clean_mask(mask)
        if np.sum(image) * np.sum(mask) == 0:
            continue
        else:
            if pid in train_ids:
                pu.save_nib_data(save_path_out+"training/", filename, image, mask, label, modality="T1")
                pu.save_img_data(save_img_path_out+"training/", filename, image, mask, label, modality="T1")
            elif pid in val_ids:
                pu.save_nib_data(save_path_out+"validation/", filename, image, mask, label, modality="T1")
                pu.save_img_data(save_img_path_out+"validation/", filename, image, mask, label, modality="T1")
            elif pid in test_ids:
                pu.save_nib_data(save_path_out+"testing/", filename, image, mask, label, modality="T1")
                pu.save_img_data(save_img_path_out+"testing/", filename, image, mask, label, modality="T1")
            else:
                print("PID Error: " + pid + "/n" + data)


def process_MSD(input_path, output_path):
    sub_root_paths = pu.sort_all_fnames(input_path)
    for sub_root_path in sub_root_paths:
        print("Processing " + pu.fname_from_path(sub_root_path) + " sub-dataset...")
        ds_info = pu.read_json(sub_root_path)

        # Create folders
        ds_name = "MSD_" + pu.fname_from_path(sub_root_path).split("_")[1]
        save_path_out, save_img_path_out = pu.create_save_folder(output_path, ds_name)
        
        # Patientwise split training/validation/testing files
        train_set = ds_info["training"][:int(len(ds_info["training"])*0.8)]
        val_set = ds_info["training"][int(len(ds_info["training"])*0.8):int(len(ds_info["training"])*0.9)]
        test_set = ds_info["training"][int(len(ds_info["training"])*0.9):]
        
        # Save Data
        pu.process_nii_json(train_set, sub_root_path, ds_info,
                            save_path_out, save_img_path_out, ds_type="training")
        pu.process_nii_json(val_set, sub_root_path, ds_info,
                            save_path_out, save_img_path_out, ds_type="validation")
        pu.process_nii_json(test_set, sub_root_path, ds_info,
                            save_path_out, save_img_path_out, ds_type="testing")
        
def process_npz_DS(input_path, output_path):
    sub_root_paths = pu.sort_all_fnames(input_path)
    for sub_root_path in sub_root_paths:
        print("Processing " + pu.fname_from_path(sub_root_path) + " sub-dataset...")
        data_name = pu.sort_all_fnames(sub_root_path)
        
        # Create folders
        ds_name = pu.fname_from_path(sub_root_path)
        save_path_out, save_img_path_out = pu.create_save_folder(output_path, ds_name)
        
        # Patientwise split training/validation/testing files
        random.shuffle(data_name)
        train_data = data_name[:int(len(data_name)*0.8)]
        val_data = data_name[int(len(data_name)*0.8):int(len(data_name)*0.9)]
        test_data = data_name[int(len(data_name)*0.9):]
        
        # Save Data
        pu.save_from_np(train_data, "training", save_path_out, save_img_path_out)
        pu.save_from_np(val_data, "validation", save_path_out, save_img_path_out)
        pu.save_from_np(test_data, "testing", save_path_out, save_img_path_out)

def process_MSK_knee(input_path, output_path):
    sub_root_paths = pu.sort_all_fnames(input_path)
    for sub_root_path in sub_root_paths:
        print("Processing " + pu.fname_from_path(sub_root_path) + " sub-dataset...")
        img_name = sorted(glob(sub_root_path + "/*_img_*.gz"))
        mask_name = sorted(glob(sub_root_path + "/*_seg_*.gz"))
        
        # Create folders
        ds_name = pu.fname_from_path(sub_root_path)
        save_path_out, save_img_path_out = pu.create_save_folder(output_path, ds_name)
        
        # Patientwise split training/validation/testing files
        training_img = img_name[:int(len(img_name)*0.8)]
        training_mask = mask_name[:int(len(mask_name)*0.8)]
        validation_img = img_name[int(len(img_name)*0.8):int(len(img_name)*0.9)]
        validation_mask = mask_name[int(len(mask_name)*0.8):int(len(mask_name)*0.9)]
        testing_img = img_name[int(len(img_name)*0.9):]
        testing_mask = mask_name[int(len(mask_name)*0.9):]
        
        # Save data
        pu.save_from_nii(training_img, 
                         training_mask, 
                         "training", 
                         save_path_out, 
                         save_img_path_out,
                         name_key="_img_pre_resampled",
                         slice_dim=0)
        pu.save_from_nii(validation_img, 
                         validation_mask, 
                         "validation", 
                         save_path_out, 
                         save_img_path_out,
                         name_key="_img_pre_resampled",
                         slice_dim=0)
        pu.save_from_nii(testing_img, 
                         testing_mask, 
                         "testing", 
                         save_path_out, 
                         save_img_path_out,
                         name_key="_img_pre_resampled",
                         slice_dim=0)

def process_PROMISE(input_path, output_path):
    # Read patient lists
    train_case_list =sorted(glob(input_path + "/training_data/*.mhd"))  #data list
    test_case_list = sorted(glob(input_path + "/test_data/*.mhd"))  #data list
    
    # Create folders
    ds_name = pu.fname_from_path(input_path)
    save_path_out, save_img_path_out = pu.create_save_folder(output_path, ds_name)
    
    # Patientwise split training/validation/testing files
    train_data = train_case_list
    val_data = test_case_list[:int(len(test_case_list)*0.5)]
    test_data = test_case_list[int(len(test_case_list)*0.5):]

    # Checking if the image and mask pairs are in the same folder.
    if len(val_data)%2 != 0:
        test_data.append(val_data[-1])
        print("Val/Test data fixed!")
        print("Val: ", len(val_data), "; Test: ", len(test_data))
    
    training_mask = sorted([x for x in train_data if "segmentation" in x])
    training_img = sorted([x for x in train_data if x not in training_mask])
    testing_mask = sorted([x for x in test_data if "segmentation" in x])
    testing_img = sorted([x for x in test_data if x not in testing_mask])
    validation_mask = sorted([x for x in val_data if "segmentation" in x])
    validation_img = sorted([x for x in val_data if x not in validation_mask])
    
    # Save data
    pu.save_from_mhd(training_img, 
                        training_mask, 
                        "training", 
                        save_path_out, 
                        save_img_path_out,
                        name_key=None,
                        slice_dim=None)
    pu.save_from_mhd(validation_img, 
                        validation_mask, 
                        "validation", 
                        save_path_out, 
                        save_img_path_out,
                        name_key=None,
                        slice_dim=None)
    pu.save_from_mhd(testing_img, 
                        testing_mask, 
                        "testing", 
                        save_path_out, 
                        save_img_path_out,
                        name_key=None,
                        slice_dim=None)
    


def process_ACDC(input_path, output_path):
    # Read patient lists
    train_patient_list = pu.sort_all_fnames(input_path + "/training/")  #data list
    test_patient_list = pu.sort_all_fnames(input_path + "/testing/")  #data list
    
    # Create folders
    ds_name = pu.fname_from_path(input_path)
    save_path_out, save_img_path_out = pu.create_save_folder(output_path, ds_name)
    
    # Patientwise split training/validation/testing files
    train_data = []
    val_data = []
    test_data = []
        # Read all data for each patient.
    for patient in train_patient_list:
        data_list = glob(patient + "/*frame*")
        train_data += data_list
    for patient in test_patient_list[:int(len(test_patient_list)*0.5)]:
        data_list = glob(patient + "/*frame*")
        val_data += data_list
    for patient in test_patient_list[int(len(test_patient_list)*0.5):]:
        data_list = glob(patient + "/*frame*")
        test_data += data_list
    
    training_mask = sorted([x for x in train_data if "gt" in x])
    training_img = sorted([x for x in train_data if x not in training_mask])
    testing_mask = sorted([x for x in test_data if "gt" in x])
    testing_img = sorted([x for x in test_data if x not in testing_mask])
    validation_mask = sorted([x for x in val_data if "gt" in x])
    validation_img = sorted([x for x in val_data if x not in validation_mask])
    
    # Save data
    pu.save_from_nii(training_img, 
                        training_mask, 
                        "training", 
                        save_path_out, 
                        save_img_path_out,
                        name_key=None,
                        slice_dim=None)
    pu.save_from_nii(validation_img, 
                        validation_mask, 
                        "validation", 
                        save_path_out, 
                        save_img_path_out,
                        name_key=None,
                        slice_dim=None)
    pu.save_from_nii(testing_img, 
                        testing_mask, 
                        "testing", 
                        save_path_out, 
                        save_img_path_out,
                        name_key=None,
                        slice_dim=None)

def porcess_CHAOS(input_path, output_path):
    # Read patient lists
    patient_list = pu.sort_all_fnames(input_path + "/Train_Sets/MR/")
    
    # Create folders
    ds_name = pu.fname_from_path(input_path)
    save_path_out_t1, save_img_path_out_t1 = pu.create_save_folder(output_path, ds_name+"_T1")
    save_path_out_t2, save_img_path_out_t2 = pu.create_save_folder(output_path, ds_name+"_T2")
    
    # Patientwise split training/validation/testing files
    train_data = patient_list[:int(len(patient_list)*0.8)]
    val_data = patient_list[int(len(patient_list)*0.8):int(len(patient_list)*0.9)]
    test_data = patient_list[int(len(patient_list)*0.9):]
    
    (t1_train_img, t1_train_mask, 
     t1_val_img, t1_val_mask, 
     t1_test_img, t1_test_mask) = ([], [], 
                                   [], [], 
                                   [], [])
    (t2_train_img, t2_train_mask, 
     t2_val_img, t2_val_mask, 
     t2_test_img, t2_test_mask) = ([], [], 
                                   [], [], 
                                   [], [])
    
    for patient in train_data:
        t1_train_img += glob(patient+"/T1DUAL/DICOM_anon/InPhase/*.dcm")
        t1_train_mask += glob(patient+"/T1DUAL/Ground/*.png")
        t2_train_img += glob(patient+"/T2SPIR/DICOM_anon/*.dcm")
        t2_train_mask += glob(patient+"/T2SPIR/Ground/*.png")
    for patient in val_data:
        t1_val_img += glob(patient+"/T1DUAL/DICOM_anon/InPhase/*.dcm")
        t1_val_mask += glob(patient+"/T1DUAL/Ground/*.png")
        t2_val_img += glob(patient+"/T2SPIR/DICOM_anon/*.dcm")
        t2_val_mask += glob(patient+"/T2SPIR/Ground/*.png")
    for patient in test_data:
        t1_test_img += glob(patient+"/T1DUAL/DICOM_anon/InPhase/*.dcm")
        t1_test_mask += glob(patient+"/T1DUAL/Ground/*.png")
        t2_test_img += glob(patient+"/T2SPIR/DICOM_anon/*.dcm")
        t2_test_mask += glob(patient+"/T2SPIR/Ground/*.png")
    
    # Save Data
    pu.save_CHAOS(t1_train_img, 
                        t1_train_mask, 
                        "training", 
                        save_path_out_t1, 
                        save_img_path_out_t1,
                        modality="T1")

    pu.save_CHAOS(t1_val_img, 
                        t1_val_mask, 
                        "validation", 
                        save_path_out_t1, 
                        save_img_path_out_t1,
                        modality="T1")
    pu.save_CHAOS(t1_test_img, 
                        t1_test_mask, 
                        "testing", 
                        save_path_out_t1, 
                        save_img_path_out_t1,
                        modality="T1") 
    pu.save_CHAOS(t2_train_img, 
                    t2_train_mask, 
                    "training", 
                    save_path_out_t2, 
                    save_img_path_out_t2,
                    modality="T2")

    pu.save_CHAOS(t2_val_img, 
                        t2_val_mask, 
                        "validation", 
                        save_path_out_t2, 
                        save_img_path_out_t2,
                        modality="T2")
    pu.save_CHAOS(t2_test_img, 
                        t2_test_mask, 
                        "testing", 
                        save_path_out_t2, 
                        save_img_path_out_t2,
                        modality="T2") 

def process_Picai(input_path, output_path):
    # Read case lists
    folder_list = ["/picai_public_images_fold0/",
                   "/picai_public_images_fold1/",
                   "/picai_public_images_fold2/",
                   "/picai_public_images_fold3/",
                   "/picai_public_images_fold4/"]
    train_img_case_list = []
    for folder in folder_list[:-1]:
        train_img_case_list += pu.sort_all_fnames(input_path + folder)
    test_img_case_list = pu.sort_all_fnames(input_path + folder_list[-1])
                
    # Create folders
    ds_name = pu.fname_from_path(input_path)
    save_path_out, save_img_path_out = pu.create_save_folder(output_path, ds_name)
    
    # Patientwise split training/validation/testing files
    mask_path = input_path + "/picai_labels/picai_labels-main/anatomical_delineations/whole_gland/AI/Guerbet23/"
    train_image_data = []
    for case in train_img_case_list:
        train_image_data += glob(case + "/*_t2w*")
    train_mask_data = []
    for img_path in train_image_data:
        img_name = pu.fname_from_path(img_path)[:13]
        train_mask_data.append(mask_path + img_name + ".nii.gz")
        
    test_image_data = []
    for case in test_img_case_list[:int(len(test_img_case_list)*0.5)]:
        test_image_data += glob(case + "/*_t2w*")
    test_mask_data = []
    for img_path in test_image_data:
        img_name = pu.fname_from_path(img_path)[:13]
        test_mask_data.append(mask_path + img_name + ".nii.gz")
        
    val_image_data = []
    for case in test_img_case_list[int(len(test_img_case_list)*0.5):]:
        val_image_data += glob(case + "/*_t2w*")
    val_mask_data = []
    for img_path in val_image_data:
        img_name = pu.fname_from_path(img_path)[:13]
        val_mask_data.append(mask_path + img_name + ".nii.gz")
    
    # Save Data
    pu.save_from_mhd(train_image_data, 
                        train_mask_data, 
                        "training", 
                        save_path_out, 
                        save_img_path_out,
                        name_key=None,
                        slice_dim=None)
    pu.save_from_mhd(val_image_data, 
                        val_mask_data, 
                        "validation", 
                        save_path_out, 
                        save_img_path_out,
                        name_key=None,
                        slice_dim=None)
    pu.save_from_mhd(test_image_data, 
                        test_mask_data, 
                        "testing", 
                        save_path_out, 
                        save_img_path_out,
                        name_key=None,
                        slice_dim=None)

def process_QUBIQ(input_path, output_path):
    # Read case lists
    pros_train_case_list = pu.sort_all_fnames(input_path + "/training_data_v3_QC/prostate/Training/")
    pros_test_case_list = pu.sort_all_fnames(input_path + "/validation_data_qubiq2021_QC/prostate/Validation/")
    kid_train_case_list = pu.sort_all_fnames(input_path + "/training_data_v3_QC/kidney/Training/")
    kid_test_case_list = pu.sort_all_fnames(input_path + "/validation_data_qubiq2021_QC/kidney/Validation/")

    # Create folders
    ds_name = pu.fname_from_path(input_path)
    save_path_out_pros, save_img_path_out_pros = pu.create_save_folder(output_path, ds_name+"_prostate")
    save_path_out_kid, save_img_path_out_kid = pu.create_save_folder(output_path, ds_name+"_kidney")

    # Patientwise split training/validation/testing files
    pros_train_data = pros_train_case_list
    pros_val_data = pros_test_case_list[:int(len(pros_test_case_list)*0.5)]
    pros_test_data = pros_test_case_list[int(len(pros_test_case_list)*0.5):]
    kid_train_data = kid_train_case_list
    kid_val_data = kid_test_case_list[:int(len(kid_test_case_list)*0.5)]
    kid_test_data = kid_test_case_list[int(len(kid_test_case_list)*0.5):]

    hyper_comb = [(pros_train_data, "training", save_path_out_pros, save_img_path_out_pros),
                  (pros_val_data, "validation", save_path_out_pros, save_img_path_out_pros),
                  (pros_test_data, "testing", save_path_out_pros, save_img_path_out_pros),
                  (kid_train_data, "training", save_path_out_kid, save_img_path_out_kid),
                  (kid_val_data, "validation", save_path_out_kid, save_img_path_out_kid),
                  (kid_test_data, "testing", save_path_out_kid, save_img_path_out_kid)]
    # Save Data
    for hyper in hyper_comb:
        pu.save_QUBIQ(hyper[0], hyper[1], hyper[2], hyper[3])


def process_HipMRI(input_path, output_path):
    # Read case lists
    img_list = pu.sort_all_fnames(input_path + "/semantic_MRs")
    mask_list = pu.sort_all_fnames(input_path + "/semantic_labels_only")
    
    # Create folders
    ds_name = pu.fname_from_path(input_path)
    save_path_out, save_img_path_out = pu.create_save_folder(output_path, ds_name)

    # Casewise split training/validation/testing files
    case_list = list(set([pu.fname_from_path(case)[:4] for case in img_list]))
    train_case_list = case_list[:int(len(case_list)*0.8)]
    val_case_list = case_list[int(len(case_list)*0.8):int(len(case_list)*0.9)]
    test_case_list = case_list[int(len(case_list)*0.9):]
    
    # Save Data
    for img_path, mask_path in tqdm(list(zip(img_list,mask_list))):
        image = pu.read_nii_file(img_path)
        mask = pu.read_nii_file(mask_path)
        
        # Delete the body label which is useless for the SAM based model segmentation.
        mask[np.where(mask==1)] = 0 # delete body label
        mask[np.where(mask==2)] = 0 # delete bone label
        filename = pu.fname_from_path(img_path)[:-7]
        if filename[:4] in train_case_list:
            pu.slice_np(image, mask, 
                        filename, "training", 
                        save_path_out, save_img_path_out,
                        split_mask=True)
        elif filename[:4] in val_case_list:
            pu.slice_np(image, mask, 
                        filename, "validation", 
                        save_path_out, save_img_path_out,
                        split_mask=True)
        elif filename[:4] in test_case_list:
            pu.slice_np(image, mask, 
                        filename, "testing", 
                        save_path_out, save_img_path_out,
                        split_mask=True)
            
def process_OAI_imorphics_dess_sag(input_path, output_path):
    # Read case lists
    image_train_list = sorted(glob(input_path + "/training/*_img_*"))
    mask_train_list = sorted(glob(input_path + "/training/*_seg_*"))
    image_test_list = sorted(glob(input_path + "/testing/*_img_*"))
    mask_test_list =  sorted(glob(input_path + "/testing/*_seg_*"))
    image_val_list =  sorted(glob(input_path + "/validation/*_img_*"))
    mask_val_list =  sorted(glob(input_path + "/validation/*_seg_*"))
    
    # Create folders
    ds_name = pu.fname_from_path(input_path)
    save_path_out, save_img_path_out = pu.create_save_folder(output_path, ds_name)
    
    # Save data
    for ds_type in ["training", "validation", "testing"]:
        if ds_type == "training":
            image_list = image_train_list
            mask_list = mask_train_list
        elif ds_type == "validation":
            image_list = image_val_list
            mask_list = mask_val_list
        elif ds_type == "testing":
            image_list = image_test_list
            mask_list = mask_test_list
        
        for img_path, mask_path in tqdm(list(zip(image_list,mask_list)), desc=ds_type+" Dataset"):
            image = pu.read_nii_file(img_path)[...,0]
            mask = pu.read_nii_file(mask_path)
            filename = "".join(pu.fname_from_path(img_path).split("_img_image"))[:-7]

            pu.slice_np(image=image, mask=mask, 
                        filename=filename, 
                        ds_type=ds_type,
                        save_nii_path_out=save_path_out,
                        save_img_path_out=save_img_path_out,
                        slice_dim=0)    
    
def process_OAI_imorphics_flash_cor(input_path, output_path):
    process_OAI_imorphics_dess_sag(input_path, output_path)

def process_OAI_imorphics_tse_sag(input_path, output_path):
    process_OAI_imorphics_dess_sag(input_path, output_path)

def process_OAIAKOA(input_path, output_path):
     # Read case lists
    image_train_list = sorted(glob(input_path + "/training/*_img*"))
    mask_train_list = sorted(glob(input_path + "/training/*_seg*"))
    image_test_list = sorted(glob(input_path + "/testing/*_img*"))
    mask_test_list =  sorted(glob(input_path + "/testing/*_seg*"))
    image_val_list =  sorted(glob(input_path + "/validation/*_img*"))
    mask_val_list =  sorted(glob(input_path + "/validation/*_seg*"))
    
    # Create folders
    ds_name = pu.fname_from_path(input_path)
    save_path_out, save_img_path_out = pu.create_save_folder(output_path, ds_name)
    
    # Save data
    for ds_type in ["training", "validation", "testing"]:
        if ds_type == "training":
            image_list = image_train_list
            mask_list = mask_train_list
        elif ds_type == "validation":
            image_list = image_val_list
            mask_list = mask_val_list
        elif ds_type == "testing":
            image_list = image_test_list
            mask_list = mask_test_list
        
        for img_path, mask_path in tqdm(list(zip(image_list,mask_list)), desc=ds_type+" Dataset"):
            image = pu.read_nii_file(img_path)
            mask = pu.read_nii_file(mask_path)
            filename = "".join(pu.fname_from_path(img_path).split("_img"))[:-7]

            pu.slice_np(image=image, mask=mask, 
                        filename=filename, 
                        ds_type=ds_type,
                        save_nii_path_out=save_path_out,
                        save_img_path_out=save_img_path_out)    
        
def process_ZIB_OAI(input_path, output_path):
    # Read case lists
    image_list = sorted(glob(input_path + "/ZIB_OAI_DESS_MRs/*.nii.gz"))
    mask_list = sorted(glob(input_path + "/ZIB_OAI_DESS_Labels/*.nii.gz"))
    
    # Create folders
    ds_name = pu.fname_from_path(input_path)
    save_path_out, save_img_path_out = pu.create_save_folder(output_path, ds_name)
    
    # Patientwise split training/validation/testing files
    image_train_list = image_list[:int(len(image_list)*0.8)]
    mask_train_list = mask_list[:int(len(mask_list)*0.8)]
    image_val_list = image_list[int(len(image_list)*0.8):int(len(image_list)*0.9)]
    mask_val_list = mask_list[int(len(mask_list)*0.8):int(len(mask_list)*0.9)]
    image_test_list = image_list[int(len(image_list)*0.9):]
    mask_test_list = mask_list[int(len(mask_list)*0.9):]
    
    # Save data
    for ds_type in ["training", "validation", "testing"]:
        if ds_type == "training":
            image_list = image_train_list
            mask_list = mask_train_list
        elif ds_type == "validation":
            image_list = image_val_list
            mask_list = mask_val_list
        elif ds_type == "testing":
            image_list = image_test_list
            mask_list = mask_test_list
            
        pu.save_from_nii(image_list, 
                         mask_list, 
                         ds_type, 
                         save_path_out, 
                         save_img_path_out)
        
def process_MSK_shoulder(input_path, output_path):
    # Read case lists
    image_list = sorted(glob(input_path + "/image_cartilages/*.hdr"))
    mask_list = sorted(glob(input_path + "/manuals_cartilages/*.nii.gz"))
    
    # Create folders
    ds_name = pu.fname_from_path(input_path)
    save_path_out, save_img_path_out = pu.create_save_folder(output_path, ds_name)

    # Patientwise split training/validation/testing files
    image_train_list = image_list[:int(len(image_list)*0.8)]
    mask_train_list = mask_list[:int(len(mask_list)*0.8)]
    image_val_list = image_list[int(len(image_list)*0.8):int(len(image_list)*0.9)]
    mask_val_list = mask_list[int(len(mask_list)*0.8):int(len(mask_list)*0.9)]
    image_test_list = image_list[int(len(image_list)*0.9):]
    mask_test_list = mask_list[int(len(mask_list)*0.9):]

    # Save data
    for ds_type in ["training", "validation", "testing"]:
        if ds_type == "training":
            image_list = image_train_list
            mask_list = mask_train_list
        elif ds_type == "validation":
            image_list = image_val_list
            mask_list = mask_val_list
        elif ds_type == "testing":
            image_list = image_test_list
            mask_list = mask_test_list
            
        pu.save_from_nii(image_list, 
                         mask_list, 
                         ds_type, 
                         save_path_out, 
                         save_img_path_out)



