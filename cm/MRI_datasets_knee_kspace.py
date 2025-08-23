# coding=utf-8
"""
MRI_datasets_knee_kspace.py

PyTorch dataloaders for FastMRI knee single-coil .h5 files, returning uniform 2D slices.
"""

from pathlib import Path
import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

# ----------------------------
# Mask Generators
# ----------------------------

def mask_1d_uniform(H, W, accel=4, acs_frac=0.04):
    """1D uniform sampling (vertical lines) + center ACS."""
    c = int(round(acs_frac * W))
    center = W // 2
    half = c // 2
    non_acs = np.ones(W, dtype=bool)
    non_acs[center - half : center - half + c] = False

    # Choose probability so E[samples] ≈ W/accel total columns (incl. ACS)
    p = max(0.0, (W / accel - c) / non_acs.sum())

    mask = np.zeros((H, W), np.float32)
    mask[:, center - half : center - half + c] = 1.0
    cols = np.random.rand(W) < p
    for j in range(W):
        if non_acs[j] and cols[j]:
            mask[:, j] = 1.0
    return mask

def mask_2d_gauss_accel(H, W, accel=8, sigma_frac=0.5):
    """2D Gaussian random sampling with target acceleration."""
    ky = np.linspace(-1, 1, H)[:, None]
    kx = np.linspace(-1, 1, W)[None, :]
    pdf = np.exp(-0.5 * ((ky / sigma_frac) ** 2 + (kx / sigma_frac) ** 2))
    pdf /= pdf.max()
    scale = (H * W / accel) / pdf.sum()
    pdf_scaled = np.clip(pdf * scale, 0, 1)
    return (np.random.rand(H, W) < pdf_scaled).astype(np.float32)

def mask_1d_gauss(H, W, accel=8, sigma_frac=0.3, acs_frac=0.04):
    """1D Gaussian-weighted (vertical) + ACS."""
    c = int(round(acs_frac * W))
    center = W // 2
    half = c // 2
    non_acs = np.ones(W, dtype=bool)
    non_acs[center - half : center - half + c] = False

    kx = np.linspace(-1, 1, W)
    pdf = np.exp(-0.5 * (kx / sigma_frac) ** 2)
    pdf /= pdf.max()

    # Scale so expected non-ACS columns ≈ W/accel - c
    denom = pdf[non_acs].sum() + 1e-8
    scale = max(0.0, (W / accel - c) / denom)
    pdf_scaled = np.clip(pdf * scale, 0, 1)

    mask = np.zeros((H, W), np.float32)
    mask[:, center - half : center - half + c] = 1.0
    r = np.random.rand(W)
    for j in range(W):
        if non_acs[j] and r[j] < pdf_scaled[j]:
            mask[:, j] = 1.0
    return mask

# ----------------------------
# Dataset
# ----------------------------

class FastMRIH5SliceDataset(Dataset):
    """Dataset of individual slices from FastMRI single-coil .h5 files, cropping/padding to a fixed shape."""
    def __init__(self, root, is_complex=False, key='reconstruction_rss', target_shape=None, sort=True, split="train"):
        root = Path(root)
        self.key = key
        self.is_complex = is_complex
        self.split = split  # "train" or "test"

        files = sorted(root.rglob('*.h5')) if sort else list(root.rglob('*.h5'))
        if not files:
            raise RuntimeError(f"No .h5 files found in {root}")

        # Build (file, slice_idx) index
        self.index = []
        for fpath in files:
            try:
                with h5py.File(fpath, 'r') as h:
                    if self.key not in h:
                        raise KeyError(f"Missing key '{self.key}' in {fpath}")
                    num_slices = h[self.key].shape[0]
            except Exception as e:
                print(f"Warning: skipping corrupted file {fpath}: {e}")
                continue
            for i in range(num_slices):
                self.index.append((fpath, i))
        if not self.index:
            raise RuntimeError(f"No valid slices found in {root}")

        # Determine target spatial shape (H, W)
        if target_shape is None:
            f0, idx0 = self.index[0]
            with h5py.File(f0, 'r') as h:
                sample = h[self.key][idx0]
            self.target_shape = sample.shape[-2:]
        else:
            self.target_shape = target_shape

        # Prepare a resize helper
        self._resize = transforms.functional.resize

        # Create and save a fixed mask ONCE for test split
        self.fixed_mask = None
        if self.split == "test":
            th, tw = self.target_shape
            # Choose your test mask here (uniform with ACS by default)
            self.fixed_mask = mask_1d_uniform(th, tw, accel=4, acs_frac=0.04).astype(np.float32)
            #self.fixed_mask = mask_1d_gauss(th, tw, accel=4, sigma_frac=0.3, acs_frac=0.04).astype(np.float32)
            #self.fixed_mask = mask_2d_gauss_accel(th, tw, accel=8, sigma_frac=0.3).astype(np.float32)

            os.makedirs("debug_masks", exist_ok=True)
            plt.imsave("debug_masks/fixed_mask.png", self.fixed_mask, cmap="gray")
            print("[Debug] Saved fixed mask to debug_masks/fixed_mask.png")

    def __len__(self):
        return len(self.index)

    @staticmethod
    def _minmax_to_pm1(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        xmin, xmax = x.min(), x.max()
        if xmax - xmin < 1e-8:
            return np.zeros_like(x, dtype=np.float32)
        x01 = (x - xmin) / (xmax - xmin)
        return x01 * 2.0 - 1.0

    def _resize_np(self, img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        """Resize a single-channel HxW numpy array using torchvision (anti-alias)."""
        t = torch.from_numpy(img).unsqueeze(0)  # [1, H, W]
        t_res = self._resize(t, [target_h, target_w], antialias=True)  # [1, th, tw]
        return t_res.squeeze(0).numpy()

    def __getitem__(self, idx):
        fpath, slice_idx = self.index[idx]
        # Load HR image (RSS)
        with h5py.File(fpath, 'r') as h:
            hr_img_orig = h['reconstruction_rss'][slice_idx].astype(np.float32)  # (H, W)

        th, tw = self.target_shape

        # Resize HR to target shape
        hr_img = self._resize_np(hr_img_orig, th, tw)

        # k-space of HR (note: fft2 assumes unshifted input)
        kspace_gt = np.fft.fftshift(np.fft.fft2(hr_img))

        H, W = hr_img.shape

        # Choose mask
        if self.split == "test":
            # Reuse the one created in __init__
            mask = self.fixed_mask.copy().astype(np.float32)
        else:
            # Training mode: sample different strategies
            p = np.random.rand()
            if p < 0.3:
                mask = mask_2d_gauss_accel(H, W, accel=8, sigma_frac=0.3)
            elif p < 0.6:
                mask = mask_1d_uniform(H, W, accel=4, acs_frac=0.04)
            else:
                mask = mask_1d_gauss(H, W, accel=4, sigma_frac=0.3, acs_frac=0.04)

        # Apply mask in k-space and reconstruct LR image (zero-filled)
        kspace_gt_masked = kspace_gt * mask
        lr_img = np.abs(np.fft.ifft2(np.fft.ifftshift(kspace_gt_masked))).astype(np.float32)

        # Ensure LR is same target size (defensive; typically already is)
        lr_img = self._resize_np(lr_img, th, tw)
        hr_inte = lr_img.copy()

        # Add channel dim [1, H, W]
        hr_img = hr_img[None, ...].astype(np.float32)
        lr_img = lr_img[None, ...].astype(np.float32)
        hr_inte = hr_inte[None, ...].astype(np.float32)
        kspace_gt_masked_final = kspace_gt_masked[None, ...].astype(np.complex64)  # [1, H, W]

        # Normalize to [-1, 1]
        def norm_pm1(x):
            return self._minmax_to_pm1(x)

        # Resize mask to (H, W) if needed (kept as float32 in [0,1])
        mask_resized = self._resize(
            torch.from_numpy(mask).unsqueeze(0), [th, tw], antialias=False
        ).squeeze(0).numpy().astype(np.float32)[None, ...]

        return {
            'lr_img': norm_pm1(lr_img),
            'hr_img': norm_pm1(hr_img),
            'hr_inte': norm_pm1(hr_inte),
            'gt': norm_pm1(hr_img),
            'mask': mask_resized,                     # [1, H, W], float32
            'kspace_gt_masked': kspace_gt_masked_final,  # [1, H, W], complex64
        }

# ----------------------------
# Dataloaders
# ----------------------------

def create_dataloader(configs, data_dir=None, evaluation=False, sort=True):
    """
    Create PyTorch DataLoader(s) for FastMRI H5 slices.
    Returns: (train_loader, test_loader)
    """
    root = data_dir or configs.data.root
    image_size = configs.data.image_size

    # Train dataset (random masks)
    train_ds = FastMRIH5SliceDataset(
        root,
        is_complex=getattr(configs.data, 'is_complex', True),
        key=getattr(configs.data, 'h5_key', 'reconstruction_rss'),
        target_shape=(image_size, image_size),
        sort=sort,
        split="train",
    )

    # Test dataset (fixed mask)
    test_ds = FastMRIH5SliceDataset(
        root,
        is_complex=getattr(configs.data, 'is_complex', True),
        key=getattr(configs.data, 'h5_key', 'reconstruction_rss'),
        target_shape=(image_size, image_size),
        sort=sort,
        split="test",
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=configs.training.batch_size,
        shuffle=not evaluation,
        num_workers=configs.training.num_workers,
        drop_last=not evaluation,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=configs.training.batch_size,
        shuffle=False,
        num_workers=configs.training.num_workers,
        drop_last=False,
        pin_memory=True,
    )

    return train_loader, test_loader
