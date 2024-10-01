import os
import glob
from ds_lib import *
from processing_utile import include

DATASETS_PATH = "R:/KNEEAI2023-A11834/Datasets/SAMRIDataSet"
PRPCESSED_ALL_PATH = "C:/DataSets/SAMRI_datasets"

dataset_lib = {
    # "Brain_Tumor_Dataset_Figshare": process_BTDF,
    # "MSD": process_MSD,
    # "amos22": process_amos22,
    # "MedSamLite": process_MSLite,
    "MSK": process_MSK,
    # "PROMISE": process_PROMISE,
    # "ACDC":process_ACDC,
    # "Brain-TR-GammaKnife-processed":process_Brain_TR_G,
    # "CHAOS_Train_Sets":porcess_CHAOS,
    # "Meningioma-SEG-CLASS":porcess_Meningioma,
    # "NCI-ISBI": process_NCI_ISBI,
    # "Picai": process_Picai,
}

# list all the dataset folders under the DATASETS_PATH folder.
ds_root = [ds for ds in glob.glob(DATASETS_PATH + "/*") if ds not in 
                                            glob.glob(DATASETS_PATH+"/*.*")]


ds_root_process = [ds for ds in ds_root if include(ds, dataset_lib.keys())]

print("\n The following datasets will be processed:")
for ds in ds_root_process:
    print(ds)
print("\n")

for ds_root_path in ds_root_process:
    ds_name = os.path.basename(ds_root_path)
    print("Processing " + ds_name +" Dataset...")
    dataset_lib[ds_name](ds_root_path, PRPCESSED_ALL_PATH)





