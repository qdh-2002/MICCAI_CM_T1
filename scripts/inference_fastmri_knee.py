import os
import torch
import argparse
import numpy as np
from torchvision.utils import save_image
from tqdm import tqdm
import torch.nn.functional as F

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from pathlib import Path
import sys
import os
# allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cm import dist_util, logger
from cm.MRI_datasets_knee_kspace import create_dataloader
from cm.script_util import model_and_diffusion_defaults, create_model_and_diffusion
from cm.karras_diffusion import karras_sample



def save_images(tensor, save_dir, prefix, start_idx):
    tensor = (tensor + 1) / 2
    tensor = tensor.clamp(0, 1)
    for i in range(tensor.size(0)):
        save_path = os.path.join(save_dir, f"{prefix}_{start_idx + i:04d}.png")
        save_image(tensor[i], save_path)

def compute_batch_metrics(pred, target):
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(pred.device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(pred.device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=False).to(pred.device)
    
    psnr_list, ssim_list, lpips_list = [], [], []
    
    for i in range(pred.shape[0]):
        # PSNR and SSIM
        psnr = psnr_metric(pred[i:i+1], target[i:i+1])
        ssim = ssim_metric(pred[i:i+1], target[i:i+1])
        
        # LPIPS: Convert grayscale to RGB and normalize to [-1, 1]
        pred_rgb = pred[i:i+1].repeat(1, 3, 1, 1)  # [1, 1, H, W] -> [1, 3, H, W]
        target_rgb = target[i:i+1].repeat(1, 3, 1, 1)  # [1, 1, H, W] -> [1, 3, H, W]
        
        # Convert from [0, 1] to [-1, 1] for LPIPS
        pred_rgb = pred_rgb * 2.0 - 1.0
        target_rgb = target_rgb * 2.0 - 1.0
        
        lpips = lpips_metric(pred_rgb, target_rgb)
        
        psnr_list.append(psnr.item())
        ssim_list.append(ssim.item())
        lpips_list.append(lpips.item())
    
    return psnr_list, ssim_list, lpips_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./samples")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=320)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--sampler", type=str, default="heun")
    parser.add_argument("--ts", type=str, default="")
    #parser.add_argument("--max_batches", type=int, default=20)
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--gauss_accel", type=int, default=8)
    parser.add_argument("--gauss_sigma", type=float, default=0.5)
    parser.add_argument("--skip", type=int, default=0)
    args = parser.parse_args()

    logger.configure()
    dist_util.setup_dist()
    os.makedirs(args.save_dir, exist_ok=True)
    interp_dir = os.path.join(args.save_dir, "interp")
    hr_dir     = os.path.join(args.save_dir, "hr")
    recon_dir  = os.path.join(args.save_dir, "recon")
    os.makedirs(interp_dir, exist_ok=True)
    os.makedirs(hr_dir,     exist_ok=True)
    os.makedirs(recon_dir,  exist_ok=True)

    # Setup dataset config
    from ml_collections import ConfigDict
    configs = ConfigDict()
    configs.data = ConfigDict({
        "root": args.data_dir,
        "mask_type": "gauss",
        "gauss_accel": args.gauss_accel,
        "gauss_sigma": args.gauss_sigma,
        "image_size": args.image_size,
        "is_complex": True,
        "magpha": False,
        "h5_key": "kspace",
        "skip": args.skip,
    })
    configs.training = ConfigDict({
        "batch_size": args.batch_size,
        "num_workers": 1
    })

    logger.log("Loading validation dataset...")
    #dataset = create_dataloader(configs, data_dir=args.data_dir, evaluation=True, sort=True)
    _, test_loader = create_dataloader(configs, data_dir=args.data_dir, evaluation=True, sort=True)

    logger.log("Creating model and diffusion...")
    model_kwargs = model_and_diffusion_defaults()
    #model_kwargs.update({"image_size": args.image_size})
    model_kwargs.update({
    "image_size": args.image_size,
    "use_fp16": args.use_fp16,
    "num_channels": 128,
    "num_res_blocks": 2,
    #"channel_mult": "1,2,4,8",
    "num_head_channels": 64,   
    "learn_sigma": False,
    "resblock_updown": True,
    "use_scale_shift_norm": True,
    "attention_resolutions": "32,16",
})
    model, diffusion = create_model_and_diffusion(**model_kwargs)
    logger.log(f"Loading model weights from: {args.model_path}")
    
    # Load state dict with strict=False to ignore PSF injection components
    # that were added during training but aren't needed for inference
    checkpoint = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)
    
    model.to(dist_util.dev())
    model.eval()
    if args.use_fp16:
        model.convert_to_fp16()

    logger.log("Starting inference...")
    all_psnr, all_ssim, all_lpips = [], [], []

    for i, batch in enumerate(tqdm(test_loader, desc="Inference Progress")):
        # if i >= args.max_batches:
        #     break

        hr      = batch["hr_img"].to(dist_util.dev())
        hr_inte = batch["hr_inte"].to(dist_util.dev())
        ts = list(range(args.steps))
        mask = batch["mask"].to(dist_util.dev())
        x_init = batch["lr_img"].to(dist_util.dev())

        with torch.no_grad():
            recon = karras_sample(
                diffusion,
                model,
                shape=hr_inte.shape,
                #sampler="multistep",
                steps=args.steps,
                model_kwargs={"hr_inte": hr_inte},
                x_init=x_init,
                #mask=mask,
                clip_denoised=True,
                device=dist_util.dev(),
                sampler=args.sampler,
                ts=ts,
                #ts=tuple(int(x) for x in args.ts.split(",")) if args.ts else None,
            )

        recon_01 = (recon.clamp(-1, 1) + 1) / 2
        hr_01    = (hr.clamp(-1, 1) + 1) / 2

        psnr_batch, ssim_batch, lpips_batch = compute_batch_metrics(recon_01, hr_01)
        all_psnr.extend(psnr_batch)
        all_ssim.extend(ssim_batch)
        all_lpips.extend(lpips_batch)

        base_idx = i * args.batch_size
        save_images(hr,      hr_dir,    "hr",    base_idx)
        save_images(hr_inte, interp_dir,"interp",base_idx)
        save_images(recon,   recon_dir, "recon", base_idx)

    mean_psnr = np.mean(all_psnr)
    mean_ssim = np.mean(all_ssim)
    mean_lpips = np.mean(all_lpips)
    
    logger.log(f"Average PSNR over {len(all_psnr)} samples: {mean_psnr:.2f} dB")
    logger.log(f"Average SSIM over {len(all_ssim)} samples: {mean_ssim:.4f}")
    logger.log(f"Average LPIPS over {len(all_lpips)} samples: {mean_lpips:.4f}")
    logger.log("Inference completed.")

if __name__ == "__main__":
    main()
