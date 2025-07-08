import os
import torch
import argparse
import numpy as np
from torchvision.utils import save_image
from tqdm import tqdm

#from torchmetrics.functional import peak_signal_noise_ratio as psnr_func
#from torchmetrics.functional import structural_similarity_index_measure as ssim_func
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


from cm import dist_util, logger
from cm.MRI_datasets_dicom_kspace import load_data_from_dicom
from cm.script_util import model_and_diffusion_defaults, create_model_and_diffusion
from cm.karras_diffusion import karras_sample


def save_images(tensor, save_dir, prefix, start_idx):
    tensor = (tensor + 1) / 2
    tensor = tensor.clamp(0, 1)
    for i in range(tensor.size(0)):
        save_path = os.path.join(save_dir, f"{prefix}_{start_idx + i:04d}.png")
        save_image(tensor[i], save_path)

def compute_batch_metrics(pred, target):
    """
    pred, target: [B, 1, H, W] in [0, 1]
    Returns: list of PSNRs, list of SSIMs
    """

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(pred.device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(pred.device)

    psnr_list = []
    ssim_list = []
    for i in range(pred.shape[0]):
        psnr = psnr_metric(pred[i:i+1], target[i:i+1])
        ssim = ssim_metric(pred[i:i+1], target[i:i+1])
        psnr_list.append(psnr.item())
        ssim_list.append(ssim.item())
    return psnr_list, ssim_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./samples")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--scan", type=str, default="AX_T2")
    parser.add_argument("--image_size", type=int, default=320)
    parser.add_argument("--crop", type=int, default=288)
    parser.add_argument("--R", type=int, default=4)
    parser.add_argument("--use_fp16", type=bool, default=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--sampler", type=str, default="heun")
    parser.add_argument("--ts", type=str, default="")
    parser.add_argument("--max_batches", type=int, default=20)
    parser.add_argument("--gauss_accel", type=int, default=8)
    parser.add_argument("--uniform_accel", type=int, default=4)
    parser.add_argument("--gauss_sigma", type=float, default=0.3)
    parser.add_argument("--calib_lines", type=int, default=9)
    parser.add_argument("--lr", type=float, default=1e-4)

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

    logger.log(f"Saving output images to: {args.save_dir}")
    logger.log("Loading validation data...")

    dataset = load_data_from_dicom(
        root_path=args.data_dir,
        batch_size=args.batch_size,
        scan=args.scan,
        scale=None,
        image_size=args.image_size,
        crop=args.crop,
        mode="val",
        select_k=300,
        select_k_start=150,
        gauss_accel=args.gauss_accel,
        uniform_accel=args.uniform_accel,
        gauss_sigma=args.gauss_sigma,
        calib_lines=args.calib_lines,
    )

    logger.log("Creating model and diffusion...")
    model_kwargs = model_and_diffusion_defaults()
    model_kwargs.update({
        "image_size": args.image_size,
        "class_cond": True,
        "use_fp16": args.use_fp16,
        "num_channels": 128,
        "num_head_channels": 64,
        "num_res_blocks": 2,
        "resblock_updown": True,
        "use_scale_shift_norm": True,
        "attention_resolutions": "32,16",
        "learn_sigma": False,
        "loss_norm": "lpips",
        "adaptive_loss": False,
        "channel_mult": "1,2,4,8",
    })

    model, diffusion = create_model_and_diffusion(**model_kwargs)
    logger.log(f"Loading model weights from: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.to(dist_util.dev())
    model.eval()
    if args.use_fp16:
        model.convert_to_fp16()

    logger.log("Starting inference...")
    all_psnr = []
    all_ssim = []

    for i, batch in enumerate(tqdm(dataset)):
        if i >= args.max_batches:
            break

        hr      = batch["hr_img"].to(dist_util.dev())        # [B,1,H,W] in [-1,1]
        hr_inte = batch["hr_inte"].to(dist_util.dev())

        with torch.no_grad():
            recon = karras_sample(
                diffusion,
                model,
                shape=hr_inte.shape,
                steps=args.steps,
                hr_inte=hr_inte,
                clip_denoised=True,
                model_kwargs={},
                device=dist_util.dev(),
                sampler=args.sampler,
                ts=tuple(int(x) for x in args.ts.split(",")) if args.ts else None,
            )
            print(f"Batch {i}: hr {hr.shape}, hr_inte {hr_inte.shape}, recon {recon.shape}")

        # Convert to [0,1] range for evaluation
        recon_01 = (recon.clamp(-1, 1) + 1) / 2
        hr_01    = (hr.clamp(-1, 1) + 1) / 2

        psnr_batch, ssim_batch = compute_batch_metrics(recon_01, hr_01)

        for b in range(len(psnr_batch)):
            idx = i * args.batch_size + b
            print(f"Sample {idx:04d} - PSNR: {psnr_batch[b]:.2f} dB | SSIM: {ssim_batch[b]:.4f}")
            all_psnr.append(psnr_batch[b])
            all_ssim.append(ssim_batch[b])

        base_idx = i * args.batch_size
        save_images(hr,      hr_dir,    "hr",    base_idx)
        save_images(hr_inte, interp_dir,"interp",base_idx)
        save_images(recon,   recon_dir, "recon", base_idx)

    mean_psnr = np.mean(all_psnr)
    mean_ssim = np.mean(all_ssim)
    logger.log(f"Average PSNR over {len(all_psnr)} samples: {mean_psnr:.2f} dB")
    logger.log(f"Average SSIM over {len(all_ssim)} samples: {mean_ssim:.4f}")
    logger.log("Inference completed.")

if __name__ == "__main__":
    main()
