import os
import numpy as np
import torch
import pydicom
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from mpi4py import MPI
from tqdm import tqdm


# ——— mask generators ———
def mask_2d_gauss(H, W, sigma_frac=0.5, accel=8):
   ky = np.linspace(-1, 1, H)[:, None]
   kx = np.linspace(-1, 1, W)[None, :]
   pdf = np.exp(-0.5 * ((ky / sigma_frac)**2 + (kx / sigma_frac)**2))
   pdf /= pdf.max()
   scale = (H * W / accel) / pdf.sum()
   pdf_scaled = np.clip(pdf * scale, 0, 1)
   return (np.random.rand(H, W) < pdf_scaled).astype(np.float32)


def mask_1d_uniform(H, W, accel=4, calib_lines=0):
   mask = np.zeros((H, W), dtype=np.float32)
   if calib_lines > 0:
       center = H // 2
       half = calib_lines // 2
       mask[center-half : center-half+calib_lines, :] = 1.0
   p = 1.0 / accel
   rows = np.random.rand(H) < p
   for i in range(H):
       if mask[i,0] == 0 and rows[i]:
           mask[i, :] = 1.0
   return mask


def mask_1d_gauss(H, W, sigma_frac=0.5, accel=8, calib_lines=0):
   ky = np.linspace(-1, 1, H)
   pdf = np.exp(-0.5 * (ky / sigma_frac)**2)
   pdf /= pdf.max()
   scale = (H / accel) / pdf.sum()
   pdf_scaled = pdf * scale
   mask = np.zeros((H, W), dtype=np.float32)
   if calib_lines > 0:
       center = H // 2
       half = calib_lines // 2
       mask[center-half : center-half+calib_lines, :] = 1.0
   r = np.random.rand(H)
   for i in range(H):
       if mask[i,0] == 0 and r[i] < pdf_scaled[i]:
           mask[i, :] = 1.0
   return mask


class MRIDicomDataset(Dataset):
   def __init__(
       self,
       root_path,
       sequence="AX_T2",
       crop=288,
       scale=4.0,
       image_size=320,
       select_k=None,
       select_k_start=None,
       calib_lines=0,
       gauss_accel=8,
       uniform_accel=4,
       gauss_sigma=0.3,
       shard=0,
       num_shards=1
   ):
       super().__init__()
       self.crop = crop
       self.scale = scale
       self.image_size = image_size
       self.calib_lines = calib_lines
       self.gauss_accel = gauss_accel
       self.uniform_accel = uniform_accel
       self.gauss_sigma = gauss_sigma
       self.sampling_map = {
           '2d_gauss': [1.0, 0.0, 0.0],
           '1d_uniform': [0.0, 1.0, 0.0],
           '1d_gauss': [0.0, 0.0, 1.0],
       }
       self.imgs = []


       print(f"Loading DICOM data from {root_path} (sequence={sequence})")
       patient_dirs = sorted(os.listdir(root_path))
       patient_dirs = [d for d in patient_dirs if os.path.isdir(os.path.join(root_path, d))]


       for pid in tqdm(patient_dirs, desc="Scanning patients"):
           seq_dirs = [d for d in os.listdir(os.path.join(root_path, pid)) if sequence in d]
           for seq in seq_dirs:
               seq_path = os.path.join(root_path, pid, seq)
               slices = []
               for fname in sorted(os.listdir(seq_path)):
                   fp = os.path.join(seq_path, fname)
                   try:
                       dcm = pydicom.dcmread(fp)
                       arr = dcm.pixel_array.astype(np.float32)
                       slices.append(arr)
                   except Exception:
                       continue
               if not slices:
                   continue
               vol = np.stack(slices, axis=0)
               vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-5)
               for img in vol:
                   t = torch.tensor(img).unsqueeze(0)
                   if t.shape[1] != image_size or t.shape[2] != image_size:
                       t = transforms.functional.resize(t, [image_size, image_size])
                   if crop is not None:
                       ch = (image_size - crop) // 2
                       t = t[:, ch:image_size-ch, :]
                   self.imgs.append(t)


       self.imgs = self.imgs[shard::num_shards]
       if select_k is not None and select_k_start is not None:
           self.imgs = self.imgs[select_k_start:select_k]
       if not self.imgs:
           raise RuntimeError("No images found.")
       print(f"Dataset ready: {len(self.imgs)} images")


   def __len__(self):
       return len(self.imgs)


   def __getitem__(self, idx):
       hr = self.imgs[idx]
       H, W = hr.shape[1], hr.shape[2]
       hr_np = hr.squeeze(0).numpy().astype(np.complex64)
       kspace = np.fft.fftshift(np.fft.fft2(hr_np))


       # 50% single-strategy, 50% mixed-strategy
       if np.random.rand() < 0.5:
           # single
           strat = np.random.choice(list(self.sampling_map.keys()))
           if strat == '2d_gauss':
               mask = mask_2d_gauss(H, W, sigma_frac=self.gauss_sigma, accel=self.gauss_accel)
           elif strat == '1d_uniform':
               mask = mask_1d_uniform(H, W, accel=self.uniform_accel, calib_lines=self.calib_lines)
           else:
               mask = mask_1d_gauss(H, W, sigma_frac=self.gauss_sigma, accel=self.gauss_accel, calib_lines=self.calib_lines)
           sampling_vec = torch.tensor(self.sampling_map[strat], dtype=torch.float32)
       else:
           # mixture of all three
           w = np.random.rand(3)
           w = w / w.sum()
           m0 = mask_2d_gauss(H, W, sigma_frac=self.gauss_sigma, accel=self.gauss_accel)
           m1 = mask_1d_uniform(H, W, accel=self.uniform_accel, calib_lines=self.calib_lines)
           m2 = mask_1d_gauss(H, W, sigma_frac=self.gauss_sigma, accel=self.gauss_accel, calib_lines=self.calib_lines)
           mask = w[0]*m0 + w[1]*m1 + w[2]*m2
           sampling_vec = torch.tensor(w.tolist(), dtype=torch.float32)


       kspace_masked = kspace * mask
       img_lr = np.abs(np.fft.ifft2(np.fft.ifftshift(kspace_masked))).astype(np.float32)
       lr = torch.from_numpy(img_lr).unsqueeze(0)


       hr = 2 * hr - 1
       lr = 2 * lr - 1


       return {
           'hr_img': hr,
           'lr_img': lr,
           'hr_inte': lr,
           'mask': torch.from_numpy(mask).unsqueeze(0),
           'sampling_vec': sampling_vec,
       }




def load_data_from_dicom(
   root_path,
   batch_size,
   scan,
   scale,
   image_size,
   crop,
   mode,
   select_k=None,
   select_k_start=0,
   calib_lines=0,
   gauss_accel=8,
   uniform_accel=4,
   gauss_sigma=0.3
):
   dataset = MRIDicomDataset(
       root_path=root_path,
       sequence=scan,
       crop=crop,
       scale=scale,
       image_size=image_size,
       select_k=select_k,
       select_k_start=select_k_start,
       calib_lines=calib_lines,
       gauss_accel=gauss_accel,
       uniform_accel=uniform_accel,
       gauss_sigma=gauss_sigma,
       shard=MPI.COMM_WORLD.Get_rank(),
       num_shards=MPI.COMM_WORLD.Get_size()
   )
   return DataLoader(
       dataset,
       batch_size=batch_size,
       shuffle=(mode=='train'),
       num_workers=4,
       drop_last=True
   )



