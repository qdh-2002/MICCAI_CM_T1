import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pydicom
from PIL import Image
from tqdm import tqdm
from mpi4py import MPI
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from cm import dist_util, logger
from cm.script_util import model_and_diffusion_defaults, create_model_and_diffusion
from cm.karras_diffusion import karras_sample


class MRIDicomDataset(Dataset):
    def __init__(
        self,
        root_path,
        sequence="AX_T2",
        crop=288,
        scale=4.0,
        image_size=320,
        select_k=None,
        shard=0,
        num_shards=1
    ):
        super().__init__()
        self.crop       = crop
        self.image_size = image_size
        self.scale      = scale
        self.lr_h       = int(crop // scale)
        self.imgs       = []

        # Walk patient folders
        patient_dirs = sorted([
            d for d in os.listdir(root_path)
            if os.path.isdir(os.path.join(root_path, d))
        ])
        print(f"Found {len(patient_dirs)} patient folders, scanning...")
        for pid in tqdm(patient_dirs, desc="Patients", unit="patient"):
            seq_dirs = [d for d in os.listdir(os.path.join(root_path, pid))
                        if sequence in d]
            for seq in seq_dirs:
                seq_path = os.path.join(root_path, pid, seq)
                # read & normalize slices
                files = sorted(os.listdir(seq_path))
                slices = []
                for fn in tqdm(files, desc=f"Reading {pid}/{seq}", leave=False, unit="slice"):
                    fp = os.path.join(seq_path, fn)
                    try:
                        ds = pydicom.dcmread(fp)
                        img = ds.pixel_array.astype(np.float32)
                        slices.append((img - img.min())/(img.max()-img.min()+1e-5))
                    except Exception:
                        continue
                if not slices:
                    continue

                volume = np.stack(slices, axis=0)  # [N, H, W]
                for sl in volume:
                    t = torch.from_numpy(sl).unsqueeze(0)  # [1, H, W]
                    # resize to square
                    if t.shape[1] != image_size or t.shape[2] != image_size:
                        t = transforms.functional.resize(
                            t, [image_size, image_size]
                        )
                    # center-crop height
                    if crop is not None and crop < image_size:
                        ch = (image_size - crop) // 2
                        t = t[:, ch:ch+crop, :]
                    # center-crop width
                    if crop is not None and crop < image_size:
                        cw = (image_size - crop) // 2
                        t = t[:, :, cw:cw+crop]
                    self.imgs.append(t)

        # sharding & optional subset
        self.imgs = self.imgs[shard::num_shards]
        if select_k is not None:
            self.imgs = self.imgs[:select_k]
        print(f"✅ Loaded {len(self.imgs)} slices total.")
        if not self.imgs:
            raise RuntimeError(f"No images found in {root_path}/{sequence}")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        hr = self.imgs[idx]  # [1, crop, crop]
        # downsample both dims to lr_h×lr_h
        lr = F.interpolate(
            hr.unsqueeze(0),
            size=(self.lr_h, self.lr_h),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)
        # upsample back to full-grid (crop×crop)
        hr_inte = F.interpolate(
            lr.unsqueeze(0),
            size=(self.crop, self.crop),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        return {
            "lr_img":  2 * lr      - 1,
            "hr_img":  2 * hr      - 1,
            "hr_inte": 2 * hr_inte - 1,
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
    train_ratio=0.8
):
    print(f"Initializing dataset: scan={scan}, crop={crop}, scale={scale}, image_size={image_size}")
    ds = MRIDicomDataset(
        root_path=root_path,
        sequence=scan,
        crop=crop,
        scale=scale,
        image_size=image_size,
        select_k=select_k,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size()
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=4,
        drop_last=True
    )
    print(f"DataLoader ready: {len(loader)} batches of size {batch_size}")
    return loader


if __name__ == "__main__":
    # quick shape-check if you run this module directly
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   required=True)
    parser.add_argument("--scan",       default="AX_DIFFUSION_ADC")
    parser.add_argument("--image_size", type=int, default=320)
    parser.add_argument("--crop",       type=int, default=288)
    parser.add_argument("--scale",      type=float, default=5.76)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--select_k",   type=int, default=4)
    args = parser.parse_args()

    loader = load_data_from_dicom(
        root_path=args.data_dir,
        batch_size=args.batch_size,
        scan=args.scan,
        scale=args.scale,
        image_size=args.image_size,
        crop=args.crop,
        mode="val",
        select_k=args.select_k
    )
    batch = next(iter(loader))
    print("lr_img   :", batch["lr_img"].shape)
    print("hr_img   :", batch["hr_img"].shape)
    print("hr_inte  :", batch["hr_inte"].shape)
