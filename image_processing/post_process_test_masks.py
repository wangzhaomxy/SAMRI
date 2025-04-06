import sys
sys.path.append("..")
from tqdm import tqdm
from utils.dataloader import *
from utils.utils import *

file_paths = TEST_IMAGE_PATH

test_dataset = NiiDataset(file_paths, multi_mask= True)

def process_masks(mask, img_path, msk_path, threshold=10):
    count = 0
    save_label = False
    mask_labels = np.unique(mask)[(np.unique(mask).nonzero()[0])]
    for label in mask_labels:
        if np.sum(mask==label) < threshold:
            count += 1
            mask[np.where(mask==label)] = 0
            save_label = True
    if np.sum(mask)==0:
        os.remove(img_path)
        os.remove(msk_path)
    elif save_label:
        new_mask = nib.Nifti1Image(mask, np.eye(4))
        nib.save(new_mask, msk_path)
    return count
    
if __name__ == "__main__":
    count_all = 0
    for _, mask in (tqdm(test_dataset)):
        img_path = test_dataset.cur_img_name
        msk_path = test_dataset.cur_msk_name
        count_all += process_masks(mask, img_path, msk_path)
    print(f"Completed! {count_all} labels have been processed!")
        
    
                
