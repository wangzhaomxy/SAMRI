from processing_utile import create_folders, sort_all_fnames, fname_from_path
from processing_utile import sort_key_fnames
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil

SOURCE_FOLDER_PATH = "C:/DataSets/SAMRI_datasets"
# SOURCE_FOLDER_PATH = "C:/DataSets/just"
TARGET_FOLDER_PATH = "C:/DataSets/SAMRI_train_test"
IMG_KEY = "_img_"
MASK_KEY = "_seg_"
RANDOM_STATE = 666


# Get folder names under the source folder.
folder_names = [fname_from_path(abs_path) + "/"
                            for abs_path in sort_all_fnames(SOURCE_FOLDER_PATH)]

# Create datasets folders under the target folder.
create_folders(TARGET_FOLDER_PATH + "/", folder_names)

# Split every dataset and copy files from source folder to target folder.
for name in tqdm(folder_names):
    source_folder = SOURCE_FOLDER_PATH + "/" + name
    target_folder = TARGET_FOLDER_PATH + "/" + name
    print(f"Processing {source_folder}")
    # Create training and testing folders under the target folder.
    create_folders(target_folder + "/", ["training/", "testing/"])
    
    # split dataset from source folder.
    img_data = sort_key_fnames(source_folder+"/processed_data", IMG_KEY)
    mask_data = sort_key_fnames(source_folder+"/processed_data", MASK_KEY)
    X_train, X_test, y_train, y_test = train_test_split(
        img_data, mask_data, train_size=0.8, random_state=RANDOM_STATE
    )
    
    # copy data to garget folder.
    for data in [X_train, y_train]:
        for file in tqdm(data):
            shutil.copy(file, target_folder+"/training")
    for data in [X_test, y_test]:
        for file in tqdm(data):
            shutil.copy(file, target_folder+"/testing")

