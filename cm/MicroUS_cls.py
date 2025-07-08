from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
import os
import pydicom
import glob
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

class MicroUSDataset(Dataset):

    def __init__(self, root_path, img_size, slice_num):
        super().__init__()
        self.data = []  
        self.h, self.w = img_size
        self.label_map = {'Positive': 1, 'Negative': 0}
        
        for label_name, label in self.label_map.items():
            # print(label_name)
            class_path = os.path.join(root_path, label_name)
            if not os.path.isdir(class_path):
                raise ValueError(f"Directory {class_path} does not exist.")
            for casename in os.listdir(class_path):
                if casename == ".DS_Store":
                    continue
                case_dir = os.path.join(class_path, casename)
                filenames = sorted(glob.glob(os.path.join(case_dir, '*.dcm')))
                filenames = self.trim_files(filenames)
                filenames = self.select_center(filenames, slice_num)
                num_files = len(filenames)
                (h_s, h_e), (w_s, w_e) = self.get_imaging_range(filenames[0])
                imgs = torch.zeros((num_files, self.h, self.w))
                for idx, filename in tqdm(enumerate(filenames), desc = 'Loading DICOMs of case ' + casename + '...', leave = False):
                    dcm = pydicom.dcmread(filename)
                    slice_data = dcm.pixel_array[h_s:h_e, w_s:w_e]
                    resized_slice = self.resize_slice(slice_data, self.h, self.w)
                    imgs[idx] = torch.tensor(2 * (resized_slice.clone() / 255) - 1, dtype=torch.float32)
                self.data.append((imgs, label))
                
    def resize_slice(self, slice_data, h, w):
        slice_tensor = torch.tensor(slice_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        resized_tensor = F.interpolate(slice_tensor, size=(h, w), mode='bicubic', align_corners=False)
        return resized_tensor.squeeze(0).squeeze(0) 
    
    def trim_files(self, filenames):
        angles_temp = []
        for filename in filenames:
            angles_temp.append(pydicom.dcmread(filename).SliceLocation)
        start_idx = angles_temp.index(max(angles_temp))
        trimmed_files = filenames[start_idx:]
        remainder = len(trimmed_files) % 4
        if remainder != 0:
            trimmed_files = trimmed_files[remainder : ]
        return trimmed_files

    def get_imaging_range(self, dicom_path):
        img = pydicom.dcmread(dicom_path).pixel_array
        h_orig, w_orig = img.shape
        h_s, h_e, w_s, w_e = 0, h_orig - 1, 0, w_orig - 1

        row_sums = img.sum(axis = 1)
        non_black_rows = np.nonzero(row_sums)[0]
        if non_black_rows.size > 0:
            h_s = non_black_rows[0]
            h_e = non_black_rows[-1] + 1

        column_sums = img.sum(axis = 0)
        non_black_columns = np.nonzero(column_sums)[0]
        if non_black_columns.size > 0:
            w_s = non_black_columns[0]
            w_e = non_black_columns[-1] + 1
        
        return (h_s, h_e), (w_s, w_e)
    
    def select_center(self, filenames, slice_num):
        """Select a fixed number of center slices from the list."""
        total_slices = len(filenames)
        if total_slices <= slice_num:
            return filenames  # Return all if fewer than slice_num
        center_idx = total_slices // 2
        half_slice = slice_num // 2
        start_idx = max(0, center_idx - half_slice)
        end_idx = start_idx + slice_num
        return filenames[start_idx:end_idx]
    
    def __len__(self):
        return len(self.data) 

    def __getitem__(self, idx):
        x, label = self.data[idx]
        return x, label