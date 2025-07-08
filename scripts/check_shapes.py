import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from cm.MRI_datasets_dicom2 import MRIDicomDataset  # make sure this import path is correct

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    required=True)
    p.add_argument("--scan",        default="AX_DIFFUSION_ADC")
    p.add_argument("--image_size",  type=int, default=320)
    p.add_argument("--crop",        type=int, default=288)
    p.add_argument("--scale",       type=float, default=5.76)
    p.add_argument("--batch_size",  type=int, default=2)
    p.add_argument("--select_k",    type=int, default=None,
                   help="Only load first K images")
    args = p.parse_args()

    # instantiate the dataset & loader
    ds = MRIDicomDataset(
        root_path=args.data_dir,
        sequence=args.scan,
        crop=args.crop,
        scale=args.scale,
        image_size=args.image_size,
        select_k=args.select_k,
        shard=0,
        num_shards=1,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=2,
    )

    # pull one batch
    batch = next(iter(loader))
    lr      = batch["lr_img"]    # (B, 1, lr_h, W_lr)
    hr      = batch["hr_img"]    # (B, 1, crop, image_size)
    hr_inte = batch["hr_inte"]   # (B, 1, image_size, image_size)

    print(">> Batch size :", lr.shape[0])
    print(">> lr_img   :", tuple(lr.shape))
    print(">> hr_img   :", tuple(hr.shape))
    print(">> hr_inte  :", tuple(hr_inte.shape))

if __name__ == "__main__":
    main()
