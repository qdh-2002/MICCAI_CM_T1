import os
import numpy as np
import torch
import pydicom
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from mpi4py import MPI
from tqdm import tqdm

def mask_1d_uniform(H, W, accel=4, calib_lines=0):
    """
    1D uniform random sampling of phase-encode lines,
    plus an optional fully-sampled calib_lines around center.
    """
    mask = np.zeros((H, W), dtype=np.float32)
    # central calibration region (if desired)
    if calib_lines > 0:
        center = H // 2
        half = calib_lines // 2
        mask[center-half : center+half, :] = 1.0

    # random remaining lines
    p = 1.0 / accel
    rows = np.random.rand(H) < p
    for i in range(H):
        if mask[i,0] == 0 and rows[i]:
            mask[i, :] = 1.0

    return mask

def cartesian_mask(
    H, W,
    R=4,
    calib_lines=24,
    var_density=True,
    sampling_method="cartesian",  # new arg
    accel=4                      # for uniform1d
):
    if sampling_method == "uniform1d":
        # ignore R, var_density; use accel and keep calib_lines
        return mask_1d_uniform(H, W, accel=accel, calib_lines=calib_lines)

    # otherwise fall back to your original Cartesian + var‐density:
    mask = np.zeros((H, W), dtype=np.float32)
    start = (W - calib_lines) // 2
    mask[:, start:start + calib_lines] = 1.0

    if var_density:
        ky = np.arange(W) - W // 2
        prob = 1.0 / (1.0 + (np.abs(ky) / (W / 2)) ** 4)
        prob /= prob.max()
        if H != W:
            # broadcast along rows
            rand = np.random.rand(H, W)
        else:
            rand = np.random.rand(W, W)
        mask[rand < prob] = 1.0
    else:
        for ky in range(W):
            if mask[0, ky] == 0 and ((ky - start) % R == 0):
                mask[:, ky] = 1.0

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
        R=4,
        calib_lines=24,
        var_density=True,
        sampling_method="uniform1d",  # default
        accel=4,                      # for uniform1d
        shard=0,
        num_shards=1
    ):
        super().__init__()
        self.crop = crop
        self.scale = scale
        self.image_size = image_size
        self.R = R
        self.calib_lines = calib_lines
        self.var_density = var_density
        self.sampling_method = sampling_method
        self.accel = accel
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
                    t = torch.tensor(img).unsqueeze(0)  # [1,H,W]
                    if t.shape[1]!=image_size or t.shape[2]!=image_size:
                        t = transforms.functional.resize(t, [image_size, image_size])
                    if crop is not None:
                        ch = (image_size - crop)//2
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
        hr = self.imgs[idx]  # [1,H,W] in [0,1]
        H, W = hr.shape[1], hr.shape[2]

        # full k-space
        hr_np = hr.squeeze(0).numpy().astype(np.complex64)
        kspace = np.fft.fftshift(np.fft.fft2(hr_np))

        # choose mask
        mask = cartesian_mask(
            H, W,
            R=self.R,
            calib_lines=self.calib_lines,
            var_density=self.var_density,
            sampling_method=self.sampling_method,
            accel=self.accel
        )
        kspace_masked = kspace * mask

        # reconstruct low-res
        img_lr = np.abs(np.fft.ifft2(np.fft.ifftshift(kspace_masked))).astype(np.float32)
        lr = torch.from_numpy(img_lr).unsqueeze(0)

        # normalize to [-1,+1]
        hr = 2*hr - 1
        lr = 2*lr - 1

        
        return {
            "hr_img":   hr,                       # target high-res
            "lr_img":   lr,                       # low-res input
            "hr_inte":  lr,                       # conditioning input
            "mask":     torch.from_numpy(mask).unsqueeze(0)
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
    R=4,
    calib_lines=24,
    var_density=True,
    sampling_method="uniform1d",
    accel=4
):
    dataset = MRIDicomDataset(
        root_path=root_path,
        sequence=scan,
        crop=crop,
        scale=scale,
        image_size=image_size,
        select_k=select_k,
        select_k_start=select_k_start,
        R=R,
        calib_lines=calib_lines,
        var_density=var_density,
        sampling_method=sampling_method,
        accel=accel,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size()
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=4,
        drop_last=True
    )
