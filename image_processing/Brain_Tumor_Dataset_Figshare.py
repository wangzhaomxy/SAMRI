import os
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import glob
import nibabel as nib

root_path = os.getcwd()
data_path = "/data/"
save_path = root_path + "/processed_data/"
label_class = {1:"meningioma", 2:"glioma", 3:"pituitary"}

data_name = sorted(glob.glob(root_path + data_path + "*"))

def read_mat_file(path):
    """
    Read the .mat file and convert the image and mask file into numpy format.
    
    Args:
        path(str): The path of a image with the format of path/xxx.mat

    Returns:
        filename(str): the strings of file name without .mat. For example: x.mat
                        will return x
        image(np.array): the numpy format of the input image.
        label(str): label classes(1,2,3) with the return of label1, label2, label3
        tumormask(np.array): the numpy format of the mask.
    """
    data = h5.File(path)
    data = data['cjdata']
    filename = os.path.basename(path)[:-4]
    # The entire dataset contains five keys: PID, image, label, tumorBorder, and
    # tumorMask. However, here we just need image, label and tumorMask for study.
    image, label, tumormask = data['image'], data['label'], data['tumorMask']
    return filename, np.array(image), "label"+str(int(np.array(label).item())),np.array(tumormask)

def save_data(path, filename, image, label, mask):
    """
    Save images and masks to .nii.gz format. Input save path, filename, image
    data, label string and mask data, save image and mask data as [1,w,h] shape.
    """
    image_name = path + filename + "_" + label + "_T2_img_0000" + ".nii.gz"
    mask_name = path + filename + "_" + label + "_A_seg_0000" + ".nii.gz"
    new_image = nib.Nifti1Image(np.expand_dims(image, axis=0), np.eye(4))
    new_mask = nib.Nifti1Image(np.expand_dims(mask, axis=0), np.eye(4))
    nib.save(new_image, image_name)
    nib.save(new_mask, mask_name)

for data in data_name:
    filename, image, label, mask = read_mat_file(data)
    save_data(save_path, filename, image, label, mask)
