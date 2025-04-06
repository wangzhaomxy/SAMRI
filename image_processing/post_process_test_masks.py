import sys
sys.path.append("..")
import os
join = os.path.join
from tqdm import tqdm
from utils.dataloader import *
from utils.utils import *
from utils.prompt import *
from glob import glob

file_paths = TEST_IMAGE_PATH
test_dataset = NiiDataset(file_paths, multi_mask= True)

count = 0
save_label = False
for embedding, mask, ori_size in (tqdm(train_dataset)):
    file_name = train_dataset.cur_name
    mask_labels = np.unique(mask)[(np.unique(mask).nonzero()[0])]
    for label in mask_labels:
        if np.sum(mask==label) < 10:
            count += 1
            mask[np.where(mask==label)] = 0
            save_label = True
    if np.sum(mask)==0:
        os.remove(file_name)
    elif save_label:
        np.savez_compressed(file_name, img=embedding, mask=mask, ori_size=ori_size)
        save_label = False
print(f"Completed! {count} labels have been processed!")
            
def process_masks():
    pass