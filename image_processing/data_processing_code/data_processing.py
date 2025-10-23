import os
from ds_lib import *
from processing_utile import include
import argparse

# ---- CLI ----
_parser = argparse.ArgumentParser(description="SAMRI data processing constants from CLI")
_parser.add_argument(
    "--dataset-path","dataset_path",
    dest="dataset_path",
    default="R:/SAMRI_DATA-Q7684/SAMRIDataSet(Raw)",
    help=f"Dataset directory.",
)
_parser.add_argument(
    "--save-path","save_path",
    dest="save_path",
    default="C:/DataSets/SAMRI_datasets",
    help=f"Embedding save directory.",
)
_args = _parser.parse_args()
# ---- CLI End----

ds_path = _args.dataset_path
save_path = _args.save_path

dataset_lib = {
    "Brain_Tumor_Dataset_Figshare": process_BTDF,
    "MSD": process_MSD,
    "Npz_dataset": process_npz_DS,
    "MSK_knee": process_MSK_knee,
    "PROMISE": process_PROMISE,
    "ACDC":process_ACDC,
    "CHAOS":porcess_CHAOS,
    "Picai": process_Picai,
    "QUBIQ": process_QUBIQ,
    "HipMRI": process_HipMRI,
    "OAI_imorphics_dess_sag": process_OAI_imorphics_dess_sag, 
    "OAI_imorphics_flash_cor": process_OAI_imorphics_flash_cor, 
    "OAI_imorphics_tse_sag": process_OAI_imorphics_tse_sag,
    "OAIAKOA" : process_OAIAKOA,
    "ZIB_OAI" : process_ZIB_OAI,
    "MSK_shoulder" : process_MSK_shoulder,
}

# list all the dataset folders under the DATASETS_PATH folder.
ds_root = [ds for ds in glob(ds_path + "/*") if ds not in 
                                            glob(ds_path+"/*.*")]

# Process all the datasets in the dataset_lib list.
ds_root_process = [ds for ds in ds_root if include(ds, dataset_lib.keys())]

print("\n The following datasets will be processed:")
for ds in ds_root_process:
    print(ds)
print("\n")

for ds_root_path in ds_root_process:
    ds_name = os.path.basename(ds_root_path)
    print("Processing " + ds_name +" Dataset...")
    dataset_lib[ds_name](ds_root_path, save_path)





