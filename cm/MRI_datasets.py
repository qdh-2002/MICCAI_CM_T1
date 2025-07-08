from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
import os
import pydicom
import glob
import math
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from .utils import *
import random
from mpi4py import MPI
import scipy

def load_data(
    root_path,
    batch_size,
    scan,
    scale,
    image_size,
    crop,
    mode,
    select_k = None
):
    image_dataset = MRIImageFolder(root_path = root_path,
                                   scan = scan,
                                   scale = scale,
                                   image_size = image_size,
                                   crop = crop,
                                   select_k = select_k,
                                   shard = MPI.COMM_WORLD.Get_rank(),
                                   num_shards = MPI.COMM_WORLD.Get_size())
    loader = DataLoader(dataset = image_dataset,
                        batch_size = batch_size,
                        shuffle = (mode == 'train'),
                        num_workers = 8,
                        drop_last = True
                        )
    return loader

def resize_bicubic(img, size):
    # img: tensor 0,1
    return transforms.ToTensor()(
        transforms.Resize(size, Image.Resampling.BICUBIC)(
            transforms.ToPILImage()(img)))

def resize_bilinear(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, Image.Resampling.BILINEAR)(
            transforms.ToPILImage()(img)))

def resize_kspace(img, size):
    H, W = img.shape[-2:]
    h_new, w_new = size
    f = torch.fft.fftshift(torch.fft.fft2(img))
    dh, dw = ((H - h_new) // 2, (W - w_new) // 2) # 288 - 50 = 238， 238 // 2 = 119, dh = 119, dw = 0
    if dh <= 0 and dw <= 0:  # upsample
        f_resized = torch.zeros((1, h_new, w_new), dtype=torch.complex64, device=img.device)
        f_resized[ : , dh : dh + h_new, dw : dw + w_new ] = f
    else:                    # downsample
        f_resized = torch.zeros((1, h_new, w_new), dtype=torch.complex64, device=img.device)
        f_resized = f[ : , (H//2 - h_new//2) : (H//2 + h_new//2), (W//2 - w_new//2) : (W//2 + w_new//2)]  
    img_ds = torch.fft.ifft2(torch.fft.ifftshift(f_resized))
    scale_factor = (h_new * w_new) / (H * W)
    img_ds = torch.real(img_ds) * scale_factor
    img_ds = img_ds.clamp(0, 1)
    return img_ds

def resize_SP(img, size, res_hr = 0.625, res_lr = 3.6, smooth = False): 
    H, W = img.shape[-2:] # 288, 320
    h_new, w_new = size  # W = w_new, 50, 320
    sp = torch.tensor(np.abs(scipy.io.loadmat('/raid/kaifengpang/SPTSR/SPTSR_data_prep/prostate_sl_profile.mat')['sl_profile']), dtype = torch.float32)
    sp_norm = sp / sp.sum()

    physical_size = H * res_hr # 288 * 0.625 = 180 mm

    z_hr = res_hr / 2 + res_hr * torch.arange(H, device = img.device) # 0.3125, 0.3125 + 0.625, 0.3125 + 0.625 * 2, ... , 0.3125 + 0.625 * 287
    z_lr = res_lr / 2 + res_lr * torch.arange(h_new, device = img.device) # 1.8, 1.8 + 3.6, 1.8 + 3.6 * 2, ... , 1.8 + 3.6 * 49

    if not smooth: 
        img_ds = torch.zeros((1, h_new, w_new), dtype = torch.float32, device = img.device)
        for i in range(h_new):
            z = z_lr[i]
            rf_start, rf_end = z - 5, z + 5 # +/- 5 mm
            idx = (z_hr >= rf_start) & (z_hr < rf_end)
            slab_lines = img[0, idx, :] 
            slab_lines = slab_lines.T.unsqueeze(0).unsqueeze(0)  # shape [1, 1, W, h_rf]
            slab_interp = F.interpolate(slab_lines, size=(W, 1000), mode='bilinear', align_corners=True) # shape [1, 1, W, 1000]
            slab_interp = slab_interp.squeeze(0).squeeze(0).T # shape [1000, W]
            img_ds[0, i, :] = (sp_norm.T @ slab_interp)[0]

    else: 
        img_ds = torch.zeros((1, H, w_new), dtype = torch.float32, device = img.device)
        for i in range(H):
            z = z_hr[i]
            rf_start, rf_end = z - 5, z + 5
            idx = (z_hr >= rf_start) & (z_hr < rf_end)
            slab_lines = img[0, idx, :] 
            slab_lines = slab_lines.T.unsqueeze(0).unsqueeze(0)  # shape [1, 1, W, h_rf]
            slab_interp = F.interpolate(slab_lines, size=(W, 1000), mode='bilinear', align_corners=True) # shape [1, 1, W, 1000]
            slab_interp = slab_interp.squeeze(0).squeeze(0).T # shape [1000, W]
            img_ds[0, i, :] = (sp_norm.T @ slab_interp)[0]

    return img_ds

class MRIImageFolder(Dataset):
    def __init__(self,
                 root_path,
                 scan, # 'tra' or 'cor'
                 scale,  
                 image_size,
                 crop = None, # crop height dim, e.g. 288
                 select_k = None,
                 shard = 0,
                 num_shards = 1
                 ):
        super().__init__()
        self.scale = scale
        self.image_size = image_size
        self.crop = crop
        self.lr_h = int(crop // scale)
        
        self.imgs = []
        # self.files = glob.glob(os.path.join(root_path, '**/*' + scan + '*'), recursive=True)
        self.files = glob.glob(os.path.join(root_path, '**', f'*{scan}*.npy'), recursive=True)
        if select_k:
            self.files = self.files[:select_k]
        for file in tqdm(self.files, desc = 'Loading MRI images...', leave = False):
            volume = np.load(file)
            volume = (volume - volume.min()) / (volume.max() - volume.min())
            h, w = volume.shape[1], volume.shape[2]
            for img in volume:
                img = torch.tensor(img).unsqueeze(0).float()
                if h != self.image_size or w != self.image_size:
                   img = resize_bicubic(img, (self.image_size, self.image_size))
                if self.crop is not None:
                    crop_h = (self.image_size - self.crop) // 2
                    img = img[ : , crop_h : self.image_size - crop_h, : ]
                self.imgs.append(img)        
        self.imgs = self.imgs[shard:][::num_shards]

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        hr = self.imgs[idx]
        # lr = resize_bicubic(hr, (self.lr_h, self.image_size))
        lr = resize_SP(hr, (self.lr_h, self.image_size))
        hr_inte = resize_bilinear(lr, (self.crop, self.image_size))

        return {'lr_img': 2 * lr - 1,
                'hr_img': 2 * hr - 1,
                'hr_inte': 2 * hr_inte - 1}
    



class MRIImageFolderThroughPlane(Dataset):
    def __init__(self,
                 root_path,
                 scan, # 'tra'
                 scale,  
                 select_k = None
                 ):
        super().__init__()
        self.scale = scale
        
        self.imgs = [] # through-plane images

        self.files = glob.glob(os.path.join(root_path, '**/*' + scan + '*'), recursive=True)
        if select_k:
            self.files = self.files[:select_k]
            
        # for file in tqdm(self.files, desc = 'Loading MRI images...', leave = False):
        #     volume_name = file.split('.')[0]
        #     volume = np.load(file) # n, h, w
        #     volume = (volume - volume.min()) / (volume.max() - volume.min())
        #     volume = volume.permute(1, 0, 2) # h, n, w
        #     for i, img in enumerate(volume):
        #         img = torch.tensor(img).unsqueeze(0).float() # 1, n, w
        #         self.imgs.append((volume_name + '_TP_' + str(i + 1), img))        

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_name, lr = self.imgs[idx]
        new_h, new_w = lr.shape[-2] * self.scale, lr.shape[-1]
        hr_inte = resize_bicubic(lr, (new_h, new_w))

        if new_h % 16 != 0:
            target_h = math.floor(new_h / 16) * 16
            padding = target_h - new_h 
            hr_inte = F.pad(hr_inte, (0, 0, 0, padding), mode="constant", value=0) 

        return {'lr_img': 2 * lr - 1,
                'hr_inte': 2 * hr_inte - 1,
                'img_name': img_name}