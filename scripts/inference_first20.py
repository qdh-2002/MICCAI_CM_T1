import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from mpi4py import MPI
import numpy as np
import pydicom

from cm import dist_util, logger
from cm.script_util import model_and_diffusion_defaults, create_model_and_diffusion
from cm.karras_diffusion import karras_sample

# --------------------------------------------
# 1) Inline DICOM Dataset + Loader
# --------------------------------------------
class MRIDicomDataset(Dataset):
    def __init__(
        self,
        root_path,
        sequence="AX_DIFFUSION_ADC",
        crop=64,
        scale=5.76,
        image_size=320,
        select_k=None,
        shard=0,
        num_shards=1,
    ):
        super().__init__()
        self.crop       = crop
        self.image_size = image_size
        self.scale      = scale
        # desired low-res height/width
        self.lr_h       = int(crop // scale)
        self.imgs       = []

        # walk patient folders
        patients = sorted([
            d for d in os.listdir(root_path)
            if os.path.isdir(os.path.join(root_path, d))
        ])
        for pid in patients:
            seq_dirs = [d for d in os.listdir(os.path.join(root_path, pid))
                        if sequence in d]
            for seq in seq_dirs:
                seq_path = os.path.join(root_path, pid, seq)
                # read all DICOM slices
                volume = []
                for fn in sorted(os.listdir(seq_path)):
                    fp = os.path.join(seq_path, fn)
                    try:
                        ds = pydicom.dcmread(fp)
                        arr = ds.pixel_array.astype(np.float32)
                        volume.append(arr)
                    except Exception:
                        continue
                if not volume:
                    continue
                vol = np.stack(volume, axis=0)
                # normalize to [0,1]
                vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-5)
                # append each slice
                for sl in vol:
                    t = torch.from_numpy(sl).unsqueeze(0)  # [1, H, W]
                    # resize to (image_size, image_size)
                    if t.shape[1] != image_size or t.shape[2] != image_size:
                        t = transforms.functional.resize(
                            t, [image_size, image_size]
                        )
                    # center-crop height to `crop`
                    if crop is not None and crop < image_size:
                        ch = (image_size - crop) // 2
                        t = t[:, ch:ch+crop, :]
                    self.imgs.append(t)

        # apply sharding
        self.imgs = self.imgs[shard::num_shards]
        # limit to first K if requested
        if select_k is not None:
            self.imgs = self.imgs[:select_k]

        if not self.imgs:
            raise RuntimeError(f"No images found in {root_path}/{sequence}")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        hr = self.imgs[idx]  # [1, crop, image_size]
        # downsample both dims to lr_h × lr_h
        lr = F.interpolate(
            hr.unsqueeze(0),
            size=(self.lr_h, self.lr_h),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        # upsample back to image_size × image_size
        hr_inte = F.interpolate(
            lr.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        return {
            "lr_img":  2 * lr      - 1,  # in [-1,1]
            "hr_img":  2 * hr      - 1,
            "hr_inte": 2 * hr_inte - 1,
        }

def load_first_k_dicom(root_path, scan, scale, image_size, crop, k):
    ds = MRIDicomDataset(
        root_path=root_path,
        sequence=scan,
        crop=crop,
        scale=scale,
        image_size=image_size,
        select_k=k,
        shard=0,
        num_shards=1,
    )
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, drop_last=False)

# --------------------------------------------
# 2) Inference Script
# --------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="First-K MRI DICOM Inference")
    parser.add_argument("--model_path",    type=str, required=True,
                        help="Path to EMA checkpoint (target_model*.pt)")
    parser.add_argument("--data_dir",      type=str, required=True,
                        help="Root DICOM directory")
    parser.add_argument("--save_dir",      type=str, default="./outputs_first20",
                        help="Where to save comparison images")
    parser.add_argument("--batch_size",    type=int, default=1,   help="Batch size for inference")
    parser.add_argument("--scan",          type=str, default="AX_DIFFUSION_ADC")
    parser.add_argument("--image_size",    type=int, default=320)
    parser.add_argument("--crop",          type=int, default=64)
    parser.add_argument("--scale",         type=float, default=5.76)
    parser.add_argument("--steps",         type=int, default=1)
    parser.add_argument("--sampler",       type=str, default="onestep",
                        choices=["onestep","heun","euler","euler_ancestral"])
    parser.add_argument("--use_fp16",      action="store_true")
    parser.add_argument("--clip_denoised", action="store_true")
    parser.add_argument("--num_images",    type=int, default=20,
                        help="Number of images to process")
    args = parser.parse_args()

    dist_util.setup_dist()
    os.makedirs(args.save_dir, exist_ok=True)
    logger.configure()

    # 1) DataLoader for first K images
    loader = load_first_k_dicom(
        root_path   = args.data_dir,
        scan        = args.scan,
        scale       = args.scale,
        image_size  = args.image_size,
        crop        = args.crop,
        k           = args.num_images,
    )

    # 2) Build & load model
    md = model_and_diffusion_defaults()
    md.update({
        "image_size": args.image_size,
        "class_cond": False,
        "use_fp16":   args.use_fp16,
    })
    model, diffusion = create_model_and_diffusion(**md)
    ckpt = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(ckpt)
    model.to(dist_util.dev()).eval()
    if args.use_fp16:
        model.convert_to_fp16()

    # 3) Run inference & save side-by-side
    for idx, batch in enumerate(loader):
        hr      = batch["hr_img"].to(dist_util.dev())
        hr_inte = batch["hr_inte"].to(dist_util.dev())

        with torch.no_grad():
            recon = karras_sample(
                diffusion, model,
                shape=hr_inte.shape,
                steps=args.steps,
                hr_inte=hr_inte,
                clip_denoised=args.clip_denoised,
                model_kwargs={},
                device=dist_util.dev(),
                sampler=args.sampler,
                ts=None,
            )

        # denorm to [0,1]
        gt   = hr.add(1).div(2).clamp(0,1)
        sr   = recon.add(1).div(2).clamp(0,1)
        # grid: [GT | SR]
        grid = make_grid(torch.cat([gt, sr], dim=0), nrow=2)
        out  = os.path.join(args.save_dir, f"cmp_{idx:02d}.png")
        save_image(grid, out)

        if idx + 1 >= args.num_images:
            break

    logger.log(f"Saved {args.num_images} comparisons to {args.save_dir}")

if __name__ == "__main__":
    main()
