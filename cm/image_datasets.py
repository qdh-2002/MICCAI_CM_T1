import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pydicom
import numpy as np
from PIL import Image
from tqdm import tqdm
from mpi4py import MPI
import torch.nn.functional as F
import random


def resize_bilinear(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, Image.Resampling.BILINEAR)(
            transforms.ToPILImage()(img))
    )


class MRIDicomDataset(Dataset):
    def __init__(self, root_dir, sequence="AX_T2", crop=288, scale=4.0, image_size=320, select_k=None,
                 mode='train', train_ratio=0.8, shard=0, num_shards=1):
        super().__init__()
        self.crop = crop
        self.image_size = image_size
        self.scale = scale
        self.lr_h = int(crop // scale)
        self.imgs = []

        print(f"🧠 Loading dataset from: {root_dir} | Mode: {mode} | Sequence: {sequence}")
        patient_dirs = sorted(os.listdir(root_dir))
        patient_dirs = [p for p in patient_dirs if os.path.isdir(os.path.join(root_dir, p))]
        print(f"🔍 Found {len(patient_dirs)} patient folders.")

        # Train-test split
        split_idx = int(len(patient_dirs) * train_ratio)
        if mode == 'train':
            selected_patients = patient_dirs[:split_idx]
        else:
            selected_patients = patient_dirs[split_idx:]

        for patient_id in tqdm(selected_patients, desc=f"📦 Loading {mode} patients"):
            patient_path = os.path.join(root_dir, patient_id)
            sequence_dirs = [
                d for d in os.listdir(patient_path)
                if sequence in d and os.path.isdir(os.path.join(patient_path, d))
            ]

            if not sequence_dirs:
                print(f"⚠️  Skipping {patient_id}: sequence folder containing '{sequence}' not found.")
                continue

            seq_dir = os.path.join(patient_path, sequence_dirs[0])  # Use the first match

            slices = []
            for f in sorted(os.listdir(seq_dir)):
                file_path = os.path.join(seq_dir, f)
                try:
                    dcm = pydicom.dcmread(file_path)
                    slices.append(dcm.pixel_array.astype(np.float32))
                except Exception as e:
                    print(f"❌ Skipping file {f}: {e}")

            if len(slices) == 0:
                print(f"⚠️  No valid slices found in {seq_dir}")
                continue

            volume = np.stack(slices, axis=0)
            volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-5)

            for img in volume:
                img = torch.tensor(img).unsqueeze(0).float()
                if img.shape[1] != self.image_size or img.shape[2] != self.image_size:
                    img = transforms.functional.resize(img, [self.image_size, self.image_size])
                if self.crop is not None:
                    crop_h = (self.image_size - self.crop) // 2
                    img = img[:, crop_h : self.image_size - crop_h, :]
                self.imgs.append(img)

        self.imgs = self.imgs[shard:][::num_shards]
        if select_k is not None:
            self.imgs = self.imgs[:select_k]

        print(f"✅ {mode.capitalize()} dataset initialized with {len(self.imgs)} total images.")

        if len(self.imgs) == 0:
            raise RuntimeError(f"No usable DICOM images found in: {root_dir} / {sequence} ({mode})")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        hr = self.imgs[idx]
        lr = F.interpolate(hr.unsqueeze(0), size=(self.lr_h, self.image_size), mode='bilinear', align_corners=True).squeeze(0)
        hr_inte = F.interpolate(lr.unsqueeze(0), size=(self.crop, self.image_size), mode='bilinear', align_corners=True).squeeze(0)
        return {
            "lr_img": 2 * lr - 1,
            "hr_img": 2 * hr - 1,
            "hr_inte": 2 * hr_inte - 1
        }


def load_data_from_dicom(
    root_path,
    batch_size,
    scan,
    scale,
    image_size,
    crop,
    mode='train',         # 'train' or 'val'
    select_k=None,
    train_ratio=0.8
):
    dataset = MRIDicomDataset(
        root_dir=root_path,
        sequence=scan,
        scale=scale,
        crop=crop,
        image_size=image_size,
        select_k=select_k,
        mode=mode,
        train_ratio=train_ratio,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size()
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=4,
        drop_last=True
    )
    return loader
