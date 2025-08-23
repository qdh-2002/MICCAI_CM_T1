"""
Based on: https://github.com/crowsonkb/k-diffusion
"""
import random
import os

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from piq import LPIPS
from torchvision.transforms import RandomCrop
from torchvision.utils import save_image, make_grid
from . import dist_util
from scipy.special import erf
from .nn import mean_flat, append_dims, append_zero
from .random_util import get_generator
from tqdm import tqdm
import math
from torch.amp import autocast, GradScaler

def compute_fourier_domain_loss(x_start_img, denoised_img, **kwargs):
    """Placeholder for Fourier domain loss"""
    return th.tensor(0.0, device=x_start_img.device)

def compute_fourier_phase_loss(x_start_img, denoised_img, **kwargs):
    """Placeholder for Fourier phase loss"""
    return th.tensor(0.0, device=x_start_img.device)

def compute_fourier_spectral_loss(x_start_img, denoised_img, **kwargs):
    """Placeholder for Fourier spectral loss"""
    return th.tensor(0.0, device=x_start_img.device)

def apply_data_consistency(denoised, kspace_gt_masked, mask, lambda_dc, step_count):
    """
    Placeholder for data consistency function.
    TODO: Implement proper data consistency enforcement.
    """
    # For now, just return the denoised image without data consistency
    return denoised

def kde_density(x: th.Tensor, eval_pts: th.Tensor, bandwidth: float) -> th.Tensor:
    ##Kernel Density Estimation with Gaussian kernel.
    # flatten to [N,1]
    x_flat = x.reshape(-1, 1)           # [N,1]
    pts    = eval_pts.reshape(1, -1)    # [1,M]
    diffs  = (x_flat - pts) / bandwidth # [N,M]
    K      = th.exp(-0.5 * diffs**2) # [N,M]
    # mean≈integral, then divided by Gaussian normalization factor
    density = K.mean(dim=0) / (bandwidth * math.sqrt(2*math.pi))
    # normalize so sum(density)≈1
    return density / (density.sum() + 1e-8)


def compute_2nd_order_histogram_loss(density_in: th.Tensor, density_out: th.Tensor) -> th.Tensor:
    """
    Compute 2nd order histogram loss using Laplace transform approach.
    
    The key insight: Laplace transform of f'(t) is sF(s) - f(0)
    For 2nd order: L{f''(t)} = s²F(s) - sf(0) - f'(0)
    
    Args:
        density_in: Input density histogram [M]
        density_out: Output density histogram [M]
        
    Returns:
        2nd order histogram loss based on Laplace transform comparison
    """
    M = len(density_in)
    device = density_in.device
    
    # Create a grid for Laplace transform variable 's'
    # Use both real and small imaginary parts for numerical stability
    s_real = th.linspace(0.1, 2.0, 16).to(device)  # Real part of s
    s_imag = th.tensor([0.01]).to(device)  # Small imaginary part for stability
    
    # Compute Laplace transforms of the density functions
    # L{f(t)} = ∫₀^∞ f(t) * e^(-st) dt
    # For discrete case: F(s) ≈ Σ f[i] * e^(-s*i*dt) * dt
    dt = 1.0 / M  # Assume uniform spacing
    t_values = th.arange(M, dtype=th.float32).to(device) * dt
    
    laplace_in_list = []
    laplace_out_list = []
    
    for s_val in s_real:
        # Complex exponential: e^(-st) = e^(-s_real*t) * e^(-i*s_imag*t)
        exp_factor = th.exp(-s_val * t_values)
        
        # Compute Laplace transforms
        laplace_in = th.sum(density_in * exp_factor) * dt
        laplace_out = th.sum(density_out * exp_factor) * dt
        
        laplace_in_list.append(laplace_in)
        laplace_out_list.append(laplace_out)
    
    laplace_in_vals = th.stack(laplace_in_list)
    laplace_out_vals = th.stack(laplace_out_list)
    
    # Compute 2nd order Laplace transforms
    # For 2nd derivative: L{f''(t)} = s²F(s) - sf(0) - f'(0)
    # Approximate f(0) and f'(0) from the discrete data
    f0_in = density_in[0]  # f(0)
    f0_out = density_out[0]
    
    # Approximate f'(0) using forward difference
    if M > 1:
        fp0_in = (density_in[1] - density_in[0]) / dt  # f'(0)
        fp0_out = (density_out[1] - density_out[0]) / dt
    else:
        fp0_in = th.tensor(0.0).to(device)
        fp0_out = th.tensor(0.0).to(device)
    
    # Compute 2nd order Laplace transforms
    s_squared = s_real * s_real
    laplace_2nd_in = s_squared * laplace_in_vals - s_real * f0_in - fp0_in
    laplace_2nd_out = s_squared * laplace_out_vals - s_real * f0_out - fp0_out
    
    # Compare the 2nd order Laplace transforms
    # Use L2 distance in the Laplace domain
    laplace_diff = laplace_2nd_in - laplace_2nd_out
    loss_2nd = th.mean(laplace_diff * laplace_diff)  # MSE in Laplace domain
    
    # Alternative: Use L1 distance for robustness
    # loss_2nd = th.mean(th.abs(laplace_diff))
    
    # Debug information
    laplace_in_norm = th.norm(laplace_2nd_in).item()
    laplace_out_norm = th.norm(laplace_2nd_out).item()
    laplace_diff_norm = th.norm(laplace_diff).item()
    
    print(f">>> DEBUG 2ND ORDER LAPLACE: laplace_in_norm={laplace_in_norm:.8f}, laplace_out_norm={laplace_out_norm:.8f}", flush=True)
    print(f">>> DEBUG 2ND ORDER LAPLACE: laplace_diff_norm={laplace_diff_norm:.8f}, loss_2nd={loss_2nd.item():.8f}", flush=True)
    
    return loss_2nd


def compute_gradients(x: th.Tensor) -> th.Tensor:
    # Compute horizontal and vertical gradients
    dx = x[..., 1:] - x[..., :-1]
    dy = x[..., :, 1:] - x[..., :, :-1]
    
    # Pad to match original size
    dx = F.pad(dx, (0, 1), mode='replicate')
    dy = F.pad(dy, (0, 1), mode='replicate')
    
    grad = th.sqrt(dx**2 + dy**2 + 1e-8)
    return grad



def get_weightings(weight_schedule, snrs, sigma_data):
    if weight_schedule == "snr":
        weightings = snrs
    elif weight_schedule == "snr+1":
        weightings = snrs + 1
    elif weight_schedule == "karras":
        weightings = snrs + 1.0 / sigma_data**2
    elif weight_schedule == "truncated-snr":
        weightings = th.clamp(snrs, min=1.0)
    elif weight_schedule == "uniform":
        weightings = th.ones_like(snrs)
    else:
        raise NotImplementedError()
    return weightings


class KarrasDenoiser:
    def __init__(
        self,
        sigma_data: float = 0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        weight_schedule="karras",
        distillation=False,
        loss_norm="lpips",
        adaptive_loss = False
    ):
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.weight_schedule = weight_schedule
        self.distillation = distillation
        self.loss_norm = loss_norm
        self.adaptive_loss = adaptive_loss
        print('Loss:', loss_norm)
        if loss_norm == "lpips":
            self.lpips_loss = LPIPS(replace_pooling=True, reduction="none")
        self.rho = rho
        self.num_timesteps = 40

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_sigmas(self, sigmas):
        return sigmas

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def training_losses(self, model, x_start, sigmas, model_kwargs=None, noise=None):
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)

        # Print sigma inspection
        import sys
        print(f">>> SIGMA VALUES INSPECTION <<<", flush=True)
        print(f">>> Sigma shape: {sigmas.shape} <<<", flush=True)
        print(f">>> Sigma min: {sigmas.min().item():.6f}, max: {sigmas.max().item():.6f} <<<", flush=True)
        print(f">>> Sigma mean: {sigmas.mean().item():.6f}, std: {sigmas.std().item():.6f} <<<", flush=True)
        print(f">>> First 5 sigmas: {sigmas[:5].cpu().numpy()} <<<", flush=True)
        print(f">>> Diffusion sigma_min: {self.sigma_min}, sigma_max: {self.sigma_max} <<<", flush=True)
        sys.stdout.flush()

        terms = {}

        dims = x_start.ndim
        x_t = x_start + noise * append_dims(sigmas, dims)
        #model_output, denoised = self.denoise(model, x_t, sigmas, **model_kwargs)
        model_output, denoised = self.denoise(
            model, x_t, sigmas,
            model_kwargs["hr_inte"],
        )

        snrs = self.get_snr(sigmas)
        weights = append_dims(
            get_weightings(self.weight_schedule, snrs, self.sigma_data), dims
        )
        terms["xs_mse"] = mean_flat((denoised - x_start) ** 2)
        terms["mse"] = mean_flat(weights * (denoised - x_start) ** 2)


        ### Remapping the pixel values to [0, 255] range, greyscale
        x_in_255 = (x_start.clamp(-1, 1) + 1) / 2 * 255.0
        x_out_255 = (denoised.clamp(-1, 1) + 1) / 2 * 255.0

        ### Define the evaluation points for KDE and bandwidth
        M = 256
        eval_pts = th.linspace(0, 255, M).to(x_start.device)  # [M]
        #bandwidth = 5.0
        pixels   = x_out_255.reshape(-1)
        N        = pixels.numel()
        sigma    = pixels.std(unbiased=True).item()
        #bandwidth= 1.06 * sigma * (N ** (-1/5))
        bandwidth = max(1.06 * sigma * (N ** (-1/5)), 1.0) 

        ### Compute the densities using KDE
        density_in = kde_density(x_in_255, eval_pts, bandwidth)  # [M]
        density_out = kde_density(x_out_255, eval_pts, bandwidth)  # [M]

        ## Compute the KDE loss (1st order)
        kde_loss = th.sum(density_in * (th.log(density_in + 1e-8) - th.log(density_out + 1e-8)))
        
        ## Compute the 2nd order histogram loss
        # Options: 'laplace', 'fourier', 'original'
        loss_method = 'fourier'  # Optimized for MRI distributions
        
        if loss_method == 'laplace':
            kde_2nd_loss = compute_2nd_order_histogram_loss(density_in, density_out)
        elif loss_method == 'fourier':
            kde_2nd_loss = compute_2nd_order_histogram_loss_fourier(density_in, density_out)
        else:  # original method - would need to implement if needed
            kde_2nd_loss = compute_2nd_order_histogram_loss(density_in, density_out)
        
        ## Compute Fourier domain losses (MRI-specific)
        # Convert from [-1, 1] back to image domain for Fourier analysis
        x_start_img = (x_start * 0.5 + 0.5).clamp(0, 1)  # Convert to [0, 1]
        denoised_img = (denoised * 0.5 + 0.5).clamp(0, 1)
        
        fourier_magnitude_loss = compute_fourier_domain_loss(x_start_img, denoised_img, 
                                                           loss_type='l1', 
                                                           weight_center=2.0, 
                                                           weight_edges=1.0)
        
        fourier_phase_loss = compute_fourier_phase_loss(x_start_img, denoised_img, weight_center=2.0)
        
        fourier_spectral_loss = compute_fourier_spectral_loss(x_start_img, denoised_img, n_bands=6)
        
        # Combine Fourier losses
        fourier_total_loss = fourier_magnitude_loss + 0.5 * fourier_phase_loss + 0.01 * fourier_spectral_loss
        
        lambda_kde = 1
        lambda_kde_2nd = 0.5  # Weight for 2nd order loss
        lambda_fourier = 0.1  # Weight for Fourier domain losses
        
        # Add losses to terms for logging
        terms["kde"] = kde_loss
        terms["kde_2nd"] = kde_2nd_loss
        terms["fourier_magnitude"] = fourier_magnitude_loss
        terms["fourier_phase"] = fourier_phase_loss
        terms["fourier_spectral"] = fourier_spectral_loss
        terms["fourier_total"] = fourier_total_loss
        terms["kde_weighted"] = kde_loss * lambda_kde
        terms["kde_2nd_weighted"] = kde_2nd_loss * lambda_kde_2nd
        terms["fourier_weighted"] = fourier_total_loss * lambda_fourier

        # Force print to terminal with flush for immediate output
        import sys
        print(f">>> KDE LOSS (1st): {kde_loss.item():.6f} <<<", flush=True)
        print(f">>> KDE LOSS (2nd): {kde_2nd_loss.item():.6f} <<<", flush=True)
        print(f">>> FOURIER MAGNITUDE: {fourier_magnitude_loss.item():.6f} <<<", flush=True)
        print(f">>> FOURIER PHASE: {fourier_phase_loss.item():.6f} <<<", flush=True)
        print(f">>> FOURIER SPECTRAL: {fourier_spectral_loss.item():.6f} <<<", flush=True)
        print(f">>> FOURIER TOTAL: {fourier_total_loss.item():.6f} <<<", flush=True)
        print(f">>> MSE LOSS: {terms['mse'].item():.6f} <<<", flush=True)
        
        total_kde = kde_loss * lambda_kde + kde_2nd_loss * lambda_kde_2nd
        total_fourier = fourier_total_loss * lambda_fourier
        print(f">>> TOTAL LOSS: {terms['mse'].item() + total_kde.item() + total_fourier.item():.6f} <<<", flush=True)
        sys.stdout.flush()  # Ensure immediate output

        if "vb" in terms:
            terms["loss"] = terms["mse"] + terms["vb"] + kde_loss * lambda_kde + kde_2nd_loss * lambda_kde_2nd + fourier_total_loss * lambda_fourier
        else:
            terms["loss"] = terms["mse"] + kde_loss * lambda_kde + kde_2nd_loss * lambda_kde_2nd + fourier_total_loss * lambda_fourier

        return terms

    def consistency_losses(
        self,
        model,
        x_start,
        num_scales,
        model_kwargs=None,
        target_model=None,
        teacher_model=None,
        teacher_diffusion=None,
        noise=None,
    ):
        hr_inte = model_kwargs.get("hr_inte") if model_kwargs else None
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)

        # Print consistency training inspection
        import sys
        print(f">>> CONSISTENCY TRAINING INSPECTION <<<", flush=True)
        print(f">>> num_scales: {num_scales} <<<", flush=True)
        print(f">>> x_start shape: {x_start.shape} <<<", flush=True)
        print(f">>> Diffusion sigma_min: {self.sigma_min}, sigma_max: {self.sigma_max} <<<", flush=True)
        sys.stdout.flush()

        dims = x_start.ndim

        def denoise_fn(x, t, hr_inte):
            return self.denoise(model, x, t, hr_inte=hr_inte)[1]

        if target_model:

            @th.no_grad()
            def target_denoise_fn(x, t, hr_inte):
                return self.denoise(target_model, x, t, hr_inte=hr_inte)[1]

        else:
            raise NotImplementedError("Must have a target model")

        if teacher_model:

            @th.no_grad()
            def teacher_denoise_fn(x, t, hr_inte):
                return teacher_diffusion.denoise(teacher_model, x, t, hr_inte=hr_inte)[1]

        @th.no_grad()
        def heun_solver(samples, t, next_t, x0, hr_inte):
            x = samples
            if teacher_model is None:
                denoiser = x0
            else:
                denoiser = teacher_denoise_fn(x, t, hr_inte)

            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)
            if teacher_model is None:
                denoiser = x0
            else:
                denoiser = teacher_denoise_fn(samples, next_t, hr_inte)

            next_d = (samples - denoiser) / append_dims(next_t, dims)
            samples = x + (d + next_d) * append_dims((next_t - t) / 2, dims)

            return samples

        @th.no_grad()
        def euler_solver(samples, t, next_t, x0, hr_inte):
            x = samples
            if teacher_model is None:
                denoiser = x0
            else:
                denoiser = teacher_denoise_fn(x, t, hr_inte)
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)

            return samples

        P_mean = -1.1
        P_std = 2.0
        max_n_scales = 1281
        ts = self.sigma_min ** (1 / self.rho) + np.arange(max_n_scales) / (num_scales - 1) * (
                    -self.sigma_min ** (1 / self.rho) + self.sigma_max ** (1 / self.rho))
        ts = ts ** self.rho
        P = erf((np.log(ts[1:]) - P_mean) / P_std / np.sqrt(2.0)) - \
            erf((np.log(ts[:-1]) - P_mean) / P_std / np.sqrt(2.0))
        P = np.where(np.arange(max_n_scales - 1) < num_scales - 1,
                     P,
                     np.zeros(max_n_scales - 1))
        P = P / np.sum(P)
        indices = np.random.choice(max_n_scales - 1, (x_start.shape[0],), True, P)

        t = th.from_numpy(ts[indices + 1]).to(x_start.device).float()  # t > t2!
        
        # Print sigma/timestep inspection for consistency training
        print(f">>> SIGMA/TIMESTEP VALUES IN CONSISTENCY TRAINING <<<", flush=True)
        print(f">>> t (sigma) shape: {t.shape} <<<", flush=True)
        print(f">>> t min: {t.min().item():.6f}, max: {t.max().item():.6f} <<<", flush=True)
        print(f">>> t mean: {t.mean().item():.6f}, std: {t.std().item():.6f} <<<", flush=True)
        print(f">>> First 5 t values: {t[:5].cpu().numpy()} <<<", flush=True)
        print(f">>> Selected indices: {indices[:5]} <<<", flush=True)
        sys.stdout.flush()
        t2 = th.from_numpy(ts[indices]).to(x_start.device).float()

        x_t = x_start + noise * append_dims(t, dims)
        
        # dropout_state = th.get_rng_state()
        seed = np.random.randint(-2**31, 2**32)
        th.manual_seed(seed)
        # with autocast(device_type='cuda'): 
        distiller = denoise_fn(x_t, t, hr_inte)

        if teacher_model is None:
            x_t2 = euler_solver(x_t, t, t2, x_start, hr_inte).detach()
        else:
            x_t2 = heun_solver(x_t, t, t2, x_start, hr_inte).detach()

        # th.set_rng_state(dropout_state)
        th.manual_seed(seed)
        # with autocast(device_type='cuda'): 
        distiller_target = target_denoise_fn(x_t2, t2, hr_inte)
        distiller_target = distiller_target.detach()


        weights = 1 / (th.abs(t2 - t) + 1e-20)

        if self.loss_norm == "l1":
            diffs = th.abs(distiller - distiller_target)
            loss = mean_flat(diffs) * weights
        elif self.loss_norm == "l2":
            diffs = (distiller - distiller_target) ** 2
            loss = mean_flat(diffs) * weights
        
        # iCM
        elif self.loss_norm == "PH-l2":
            # with autocast(device_type='cuda'):
            c = 0.00054 * 320
            losses = (distiller - distiller_target) ** 2
            losses = losses.reshape(losses.shape[0], -1)
            data_dim = losses.shape[-1]
            losses = th.sqrt(th.sum(losses, axis=-1) + c ** 2) - c
            losses = losses / np.sqrt(data_dim)
            loss = losses * weights
            # diffs = th.sqrt((distiller - distiller_target) ** 2 + c**2) - c
            # loss = mean_flat(diffs) * weights

        elif self.loss_norm == "l2-32":
            distiller = F.interpolate(distiller, size=32, mode="bilinear")
            distiller_target = F.interpolate(
                distiller_target,
                size=32,
                mode="bilinear",
            )
            diffs = (distiller - distiller_target) ** 2
            loss = mean_flat(diffs) * weights
        elif self.loss_norm == "lpips":
            if x_start.shape[-1] < 256:
                distiller = F.interpolate(distiller, size=320, mode="bilinear")
                distiller_target = F.interpolate(
                    distiller_target, size=320, mode="bilinear"
                )

            loss = (
                self.lpips_loss(
                    (distiller + 1) / 2.0,
                    (distiller_target + 1) / 2.0,
                )
                * weights
            )
        else:
            raise ValueError(f"Unknown loss norm {self.loss_norm}")

        ### Remapping the pixel values to [0, 255] range, greyscale
        x_in_255 = (x_start.clamp(-1, 1) + 1) / 2 * 255.0
        x_out_255 = (distiller.clamp(-1, 1) + 1) / 2 * 255.0

        ### Define the evaluation points for KDE and bandwidth
        M = 256
        eval_pts = th.linspace(0, 255, M).to(x_start.device)  # [M]
        pixels   = x_out_255.reshape(-1)
        N        = pixels.numel()
        sigma    = pixels.std(unbiased=True).item()
        bandwidth = max(1.06 * sigma * (N ** (-1/5)), 1.0) 

        ### Compute the densities using KDE
        density_in = kde_density(x_in_255, eval_pts, bandwidth)  # [M]
        density_out = kde_density(x_out_255, eval_pts, bandwidth)  # [M]

        ## Compute the KDE loss (1st order)
        kde_loss = th.sum(density_in * (th.log(density_in + 1e-8) - th.log(density_out + 1e-8)))
        
        ## Compute the 2nd order histogram loss
        # Options: 'laplace', 'fourier', 'original'
        # For MRI: Fourier method is recommended due to better handling of
        # complex-valued nature, frequency domain relevance, and numerical stability
        loss_method = 'fourier'  # Optimized for MRI distributions
        
        if loss_method == 'laplace':
            kde_2nd_loss = compute_2nd_order_histogram_loss(density_in, density_out)
        elif loss_method == 'fourier':
            kde_2nd_loss = compute_2nd_order_histogram_loss_fourier(density_in, density_out)
        else:  # original method - would need to implement if needed
            kde_2nd_loss = compute_2nd_order_histogram_loss(density_in, density_out)
        


        lambda_kde = 0.01  
        lambda_kde_2nd = 0.005

        # Print losses in consistency training
        print(f">>> CONSISTENCY KDE LOSS (1st): {kde_loss.item():.6f} ", flush=True)
        print(f">>> CONSISTENCY KDE LOSS (2nd): {kde_2nd_loss.item():.6f} ", flush=True)
        print(f">>> CONSISTENCY KDE RATIO (2nd/1st): {(kde_2nd_loss.item() / max(kde_loss.item(), 1e-8)):.6f} ", flush=True)
        print(f">>> CONSISTENCY BASE LOSS: {loss.mean().item():.6f} ", flush=True)
        

        kde_weighted = kde_loss * lambda_kde 
        kde_2nd_weighted = kde_2nd_loss * lambda_kde_2nd 
        
        loss_with_all = loss + kde_weighted + kde_2nd_weighted
        print(f">>> CONSISTENCY TOTAL LOSS: {loss_with_all.mean().item():.6f} <<<", flush=True)
        sys.stdout.flush()

        # --------------------------------------------------------------------
        # Return the scalar loss plus the three intermediate tensors:
        #   xt      = original noisy input
        #   x0_pred = model's denoised prediction
        #   xs      = list of next-step samples (here just x_t2)
        # --------------------------------------------------------------------
        return {
            "loss":    loss_with_all,
            "kde":     kde_loss,
            "kde_2nd": kde_2nd_loss,
            "kde_weighted": kde_weighted,
            "kde_2nd_weighted": kde_2nd_weighted,
            "base_loss": loss,
            "xt":      x_t,
            "x0_pred": distiller,
            "xs":      [x_t2],
        }


    def progdist_losses(
        self,
        model,
        x_start,
        num_scales,
        model_kwargs=None,
        teacher_model=None,
        teacher_diffusion=None,
        noise=None,
    ):
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)

        dims = x_start.ndim

        def denoise_fn(x, t):
            return self.denoise(model, x, t, **model_kwargs)[1]

        @th.no_grad()
        def teacher_denoise_fn(x, t):
            return teacher_diffusion.denoise(teacher_model, x, t, **model_kwargs)[1]

        @th.no_grad()
        def euler_solver(samples, t, next_t):
            x = samples
            denoiser = teacher_denoise_fn(x, t)
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)

            return samples

        @th.no_grad()
        def euler_to_denoiser(x_t, t, x_next_t, next_t):
            denoiser = x_t - append_dims(t, dims) * (x_next_t - x_t) / append_dims(
                next_t - t, dims
            )
            return denoiser

        indices = th.randint(0, num_scales, (x_start.shape[0],), device=x_start.device)

        t = self.sigma_max ** (1 / self.rho) + indices / num_scales * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        t2 = self.sigma_max ** (1 / self.rho) + (indices + 0.5) / num_scales * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho

        t3 = self.sigma_max ** (1 / self.rho) + (indices + 1) / num_scales * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t3 = t3**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        denoised_x = denoise_fn(x_t, t)

        x_t2 = euler_solver(x_t, t, t2).detach()
        x_t3 = euler_solver(x_t2, t2, t3).detach()

        target_x = euler_to_denoiser(x_t, t, x_t3, t3).detach()

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)
        if self.loss_norm == "l1":
            diffs = th.abs(denoised_x - target_x)
            loss = mean_flat(diffs) * weights
        elif self.loss_norm == "l2":
            diffs = (denoised_x - target_x) ** 2
            loss = mean_flat(diffs) * weights
        elif self.loss_norm == "lpips":
            if x_start.shape[-1] < 256:
                denoised_x = F.interpolate(denoised_x, size=224, mode="bilinear")
                target_x = F.interpolate(target_x, size=224, mode="bilinear")
            loss = (
                self.lpips_loss(
                    (denoised_x + 1) / 2.0,
                    (target_x + 1) / 2.0,
                )
                * weights
            )
        else:
            raise ValueError(f"Unknown loss norm {self.loss_norm}")

        terms = {}
        terms["loss"] = loss

        return terms

    def denoise(self, model, x_t, sigmas, **model_kwargs):
        import torch.distributed as dist

        if not self.distillation:
            c_skip, c_out, c_in = [
                append_dims(x, x_t.ndim) for x in self.get_scalings(sigmas)
            ]
        else:
            c_skip, c_out, c_in = [
                append_dims(x, x_t.ndim).to(x_t.dtype)
                for x in self.get_scalings_for_boundary_condition(sigmas)
            ]
        rescaled_t = (1000 * 0.25 * th.log(sigmas + 1e-44)).to(x_t.dtype)
        model_output = model(c_in * x_t, rescaled_t, **model_kwargs)
        denoised = c_out * model_output + c_skip * x_t
        return model_output, denoised

    def save_training_intermediates_method(self, x_start, x_t, t, distiller=None):
        """
        Save intermediate training results showing noisy images at different noise levels.
        
        Args:
            x_start: Clean ground truth images [B, C, H, W]
            x_t: Noisy images at noise level t [B, C, H, W]  
            t: Noise levels [B]
            distiller: Model predictions (optional) [B, C, H, W]
        """
        
        # Create debug directory
        debug_dir = './debug_training_intermediates'
        os.makedirs(debug_dir, exist_ok=True)
        
        # Only save first batch element to avoid too many files
        batch_idx = 0
        
        # Convert to [0,1] range for visualization
        def normalize_for_save(tensor):
            tensor = tensor[batch_idx:batch_idx+1]  # Take first element
            return (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
        
        step = self.training_step_counter
        noise_level = t[batch_idx].item() if len(t.shape) > 0 else t.item()
        
        # Save clean image
        x_start_norm = normalize_for_save(x_start)
        save_image(x_start_norm, os.path.join(debug_dir, f'step_{step:06d}_noise_{noise_level:.3f}_clean.png'))
        
        # Save noisy image
        x_t_norm = normalize_for_save(x_t)
        save_image(x_t_norm, os.path.join(debug_dir, f'step_{step:06d}_noise_{noise_level:.3f}_noisy.png'))
        
        # Save model prediction if available
        if distiller is not None:
            distiller_norm = normalize_for_save(distiller)
            save_image(distiller_norm, os.path.join(debug_dir, f'step_{step:06d}_noise_{noise_level:.3f}_pred.png'))
        
        # Create a grid showing noise level progression every 1000 steps
        if step % 1000 == 0:
            self.save_noise_level_grid(x_start[batch_idx:batch_idx+1], step)
        
        print(f"Saved training intermediates at step {step}, noise level {noise_level:.3f}")

    def save_noise_level_grid(self, x_clean, step):
        """
        Create a grid showing the same image with different noise levels.
        """
        
        debug_dir = './debug_training_intermediates'
        
        # Create noise levels from min to max
        noise_levels = th.linspace(self.sigma_min, self.sigma_max, 8, device=x_clean.device)
        noisy_images = []
        
        for sigma in noise_levels:
            noise = th.randn_like(x_clean)
            x_noisy = x_clean + noise * sigma
            # Normalize for visualization
            x_noisy_norm = (x_noisy - x_noisy.min()) / (x_noisy.max() - x_noisy.min() + 1e-8)
            noisy_images.append(x_noisy_norm)
        
        # Concatenate all images into a single tensor for make_grid
        # Each image should be [1, C, H, W], concatenate along batch dimension
        all_images = th.cat(noisy_images, dim=0)  # Shape: [8, C, H, W]
        
        # Create grid
        grid = make_grid(all_images, nrow=4, padding=2, normalize=False)
        save_image(grid, os.path.join(debug_dir, f'step_{step:06d}_noise_progression.png'))
        
        print(f"Saved noise level progression grid at step {step}")

def karras_sample(
    diffusion,
    model,
    shape,
    steps,
    x_init,
    clip_denoised=True,
    progress=False,
    callback=None,
    model_kwargs=None,
    device=None,
    sigma_min=0.002,
    sigma_max=80.0,  # higher for highres?
    rho=7.0,
    sampler="heun",
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    generator=None,
    ts=None,
):
    if generator is None:
        generator = get_generator("dummy")

    if sampler == "progdist":
        sigmas = get_sigmas_karras(steps + 1, sigma_min, sigma_max, rho, device=device)
    else:
        sigmas = get_sigmas_karras(steps, sigma_min, sigma_max, rho, device=device)
    
    # Ensure sigmas are in the same dtype as x_init
    sigmas = sigmas.to(x_init.dtype)

    x_T = generator.randn(*shape, device=device) * sigma_max
    skip = 0

    #sigma_k = sigmas[skip]
    sigma_x_init = 1.0
    #x_T = x_init

    noise_init = th.randn_like(x_init)
    dims_init = x_init.ndim
    # Convert sigma_x_init to tensor to work with append_dims
    sigma_x_init_tensor = th.tensor(sigma_x_init, device=x_init.device, dtype=x_init.dtype)
    x_T = x_init + noise_init * append_dims(sigma_x_init_tensor, dims_init)

    B,C,H,W = x_init.shape
    n = C*H*W


    sample_fn = {
        "heun": sample_heun,
        "dpm": sample_dpm,
        "ancestral": sample_euler_ancestral,
        "onestep": sample_onestep,
        "progdist": sample_progdist,
        "euler": sample_euler,
        "multistep": stochastic_iterative_sampler,
    }[sampler]

    if sampler in ["heun", "dpm"]:
        sampler_args = dict(
            s_churn=s_churn, s_tmin=s_tmin, s_tmax=s_tmax, s_noise=s_noise
        )
    elif sampler == "multistep":
        ts = list(range(skip, steps))
        sigmas = sigmas[skip:]
        sampler_args = dict(
            ts=ts, t_min=sigma_min, t_max=sigma_max, rho=diffusion.rho, steps=steps
        )
    else:
        sampler_args = {}

    def denoiser(x_t, sigma):
        # Ensure sigma has the same dtype as x_t
        if hasattr(sigma, 'dtype') and sigma.dtype != x_t.dtype:
            sigma = sigma.to(x_t.dtype)
        

        filtered_model_kwargs = {}
        if model_kwargs is not None:
            for key, value in model_kwargs.items():
                if key not in ['skip', 'x_init', 'lambda_dc', 'kspace_gt_masked']:
                    filtered_model_kwargs[key] = value
        
        _, denoised = diffusion.denoise(model, x_t, sigma, **filtered_model_kwargs)
        if clip_denoised:
            denoised = denoised.clamp(-1, 1)
        
        # Apply data consistency only in the first denoising step
        denoised_before_dc = denoised.clone() if (model_kwargs is not None and 'mask' in model_kwargs and 'kspace_gt_masked' in model_kwargs and denoiser.step_count < 10) else None
        
        if (model_kwargs is not None and 'mask' in model_kwargs and 'kspace_gt_masked' in model_kwargs 
            and denoiser.step_count < 10):
            mask = model_kwargs['mask']
            kspace_gt_masked = model_kwargs['kspace_gt_masked']
            
            # Apply data consistency with configurable strength
            lambda_dc = model_kwargs.get('lambda_dc', 1.0)
            denoised = apply_data_consistency(denoised, kspace_gt_masked, mask, lambda_dc, denoiser.step_count)
            
            # Ensure we still clip after data consistency
            if clip_denoised:
                denoised = denoised.clamp(-1, 1)
        
        # Save intermediate results after every denoising step
        debug_dir = './debug_denoising_steps'
        os.makedirs(debug_dir, exist_ok=True)
        
        # Convert sigma to scalar for filename
        sigma_val = float(sigma.mean()) if hasattr(sigma, 'mean') else float(sigma)
        
        # Save input (noisy image)
        x_t_norm = (x_t.clamp(-1, 1) + 1) * 0.5  # Convert from [-1,1] to [0,1]
        save_image(x_t_norm, os.path.join(debug_dir, f'step_{denoiser.step_count:03d}_sigma_{sigma_val:.6f}_input.png'))
        
        # Save denoised output
        denoised_norm = (denoised.clamp(-1, 1) + 1) * 0.5  # Convert from [-1,1] to [0,1]
        save_image(denoised_norm, os.path.join(debug_dir, f'step_{denoiser.step_count:03d}_sigma_{sigma_val:.6f}_denoised.png'))
        
        # If data consistency was applied, save the before/after comparison
        if denoised_before_dc is not None:
            denoised_before_dc_norm = (denoised_before_dc.clamp(-1, 1) + 1) * 0.5
            save_image(denoised_before_dc_norm, os.path.join(debug_dir, f'step_{denoiser.step_count:03d}_sigma_{sigma_val:.6f}_before_dc.png'))
            save_image(denoised_norm, os.path.join(debug_dir, f'step_{denoiser.step_count:03d}_sigma_{sigma_val:.6f}_after_dc.png'))
        
        print(f"Saved denoising step {denoiser.step_count}, sigma={sigma_val:.6f} to {debug_dir}")
        
        # Increment step counter
        denoiser.step_count += 1
        
        return denoised

    # Initialize step counter for data consistency after function definition
    denoiser.step_count = 0

    # x_0 = sample_fn(
    #     denoiser,
    #     x_T,
    #     sigmas,
    #     generator,
    #     progress=progress,
    #     callback=callback,
    #     **sampler_args,
    # )
    # return x_0.clamp(-1, 1)

    if sampler == "onestep":
        # sample_onestep(distiller, x, sigmas, hr_inte)
        x_0 = sample_onestep(
            denoiser,
            x_T,
            sigmas,
        )
    else:
        # all other samplers expect: fn(distiller, x, sigmas, hr_inte, generator, progress, callback, **sampler_args)
        x_0 = sample_fn(
            denoiser,
            x_T,
            sigmas,
            generator,
            progress=progress,
            callback=callback,
            **sampler_args,
        )
        
    return x_0.clamp(-1, 1)


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = th.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)


def get_ancestral_step(sigma_from, sigma_to):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    sigma_up = (
        sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2
    ) ** 0.5
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


@th.no_grad()
def sample_euler_ancestral(model, x, sigmas, generator, progress=False, callback=None):
    """Ancestral sampling with Euler method steps."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        denoised = model(x, sigmas[i] * s_in)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1])
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised,
                }
            )
        d = to_d(x, sigmas[i], denoised)
        # Euler method
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        x = x + generator.randn_like(x) * sigma_up
    return x


@th.no_grad()
def sample_midpoint_ancestral(model, x, ts, generator, progress=False, callback=None):
    """Ancestral sampling with midpoint method steps."""
    s_in = x.new_ones([x.shape[0]])
    step_size = 1 / len(ts)
    if progress:
        from tqdm.auto import tqdm

        ts = tqdm(ts)

    for tn in ts:
        dn = model(x, tn * s_in)
        dn_2 = model(x + (step_size / 2) * dn, (tn + step_size / 2) * s_in)
        x = x + step_size * dn_2
        if callback is not None:
            callback({"x": x, "tn": tn, "dn": dn, "dn_2": dn_2})
    return x


@th.no_grad()
def sample_heun(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        eps = generator.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = denoiser(x, sigma_hat * s_in)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            # Euler method
            x = x + d * dt
        else:
            # Heun's method
            x_2 = x + d * dt
            denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
    return x


@th.no_grad()
def sample_euler(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        sigma = sigmas[i]
        denoised = denoiser(x, sigma * s_in)
        d = to_d(x, sigma, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "denoised": denoised,
                }
            )
        dt = sigmas[i + 1] - sigma
        x = x + d * dt
    return x


@th.no_grad()
def sample_dpm(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
):
    """A sampler inspired by DPM-Solver-2 and Algorithm 2 from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        eps = generator.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = denoiser(x, sigma_hat * s_in)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )
        # Midpoint method, where the midpoint is chosen according to a rho=3 Karras schedule
        sigma_mid = ((sigma_hat ** (1 / 3) + sigmas[i + 1] ** (1 / 3)) / 2) ** 3
        dt_1 = sigma_mid - sigma_hat
        dt_2 = sigmas[i + 1] - sigma_hat
        x_2 = x + d * dt_1
        denoised_2 = denoiser(x_2, sigma_mid * s_in)
        d_2 = to_d(x_2, sigma_mid, denoised_2)
        x = x + d_2 * dt_2
    return x


@th.no_grad()
def sample_onestep(
    distiller,
    x,
    sigmas,
    generator=None,
    progress=False,
    callback=None,
):
    """Single-step generation from a distilled model."""
    s_in = x.new_ones([x.shape[0]])
    # Ensure sigma and s_in have compatible dtypes
    sigma_scaled = (sigmas[0] * s_in).to(x.dtype)

    return distiller(x, sigma_scaled)


@th.no_grad()
def stochastic_iterative_sampler(
    distiller,
    x,
    sigmas,
    generator,
    ts,
    progress=False,
    callback=None,
    t_min=0.002,
    t_max=40.0,
    rho=7.0,
    steps=40,
):
    print(f"DEBUG: stochastic_iterative_sampler called!")
    print(f"DEBUG: ts = {ts}")
    print(f"DEBUG: steps = {steps}")
    
    # Handle case where ts is None - generate time steps
    if ts is None:
        ts = list(range(steps))
        print(f"DEBUG: Generated ts = {ts}")
    
    print(f"DEBUG: len(ts) = {len(ts)}")
    print(f"DEBUG: Loop will run {len(ts) - 1} iterations")
    
    # Setup comprehensive intermediate saving
    debug_dir = './debug_stochastic_iterative_steps'
    os.makedirs(debug_dir, exist_ok=True)
    
    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])

    for i in range(len(ts) - 1):
        print(f"\n=== Stochastic Iterative Step {i+1}/{len(ts)-1} ===")
        
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        
        # Save input state before denoising
        x_input_norm = (x.clamp(-1, 1) + 1) * 0.5  # Convert from [-1,1] to [0,1]
        save_image(x_input_norm, os.path.join(debug_dir, f'step_{i:03d}_input_t_{float(t):.6f}.png'))
        
        sigma_x_init = 1.2
        t_adjusted = sigma_x_init
        t_adjusted = t

        if i == 0:
            print(f"t_adjusted * s_in (first iteration): {t_adjusted * s_in}")
            x0 = distiller(x, t_adjusted * s_in)
            print(f"Used t_adjusted={t_adjusted} for first iteration")
        else:
            print(f"t * s_in (iteration {i}): {t * s_in}")
            x0 = distiller(x, t * s_in)
            print(f"Used t={float(t):.6f} for iteration {i}")

        # Save denoised result x0
        x0_norm = (x0.clamp(-1, 1) + 1) * 0.5  # Convert from [-1,1] to [0,1]
        save_image(x0_norm, os.path.join(debug_dir, f'step_{i:03d}_denoised_t_{float(t):.6f}.png'))

        # Calculate next noise level
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        
        print(f"Next noise level: {float(next_t):.6f}")
        
        # Add noise for next iteration
        noise_amount = np.sqrt(next_t**2 - t_min**2)
        noise = generator.randn_like(x) * noise_amount
        x = x0 + noise
        
        # Save final state after noise addition
        x_final_norm = (x.clamp(-1, 1) + 1) * 0.5
        save_image(x_final_norm, os.path.join(debug_dir, f'step_{i:03d}_after_noise_next_t_{float(next_t):.6f}.png'))
        
        print(f"Added noise with amount: {float(noise_amount):.6f}")
        print(f"Saved all intermediate results for step {i} to {debug_dir}")
        
        # Save a comparison grid showing the progression
        if i < 10:  # Only for first 10 steps to avoid too many files
            try:
                from torchvision.utils import make_grid
                comparison_imgs = [
                    x_input_norm[0:1],  # Input 
                    x0_norm[0:1],       # Denoised
                    x_final_norm[0:1]   # After noise
                ]
                grid = make_grid(th.cat(comparison_imgs, dim=0), nrow=3, padding=2)
                save_image(grid, os.path.join(debug_dir, f'step_{i:03d}_comparison_grid.png'))
            except Exception as e:
                print(f"Could not save comparison grid: {e}")

    print(f"\n=== Stochastic Iterative Sampling Complete ===")
    print(f"Final result saved with all {len(ts)-1} intermediate steps")
    return x


@th.no_grad()
def sample_progdist(
    denoiser,
    x,
    sigmas,
    generator=None,
    progress=False,
    callback=None,
):
    s_in = x.new_ones([x.shape[0]])
    sigmas = sigmas[:-1]  # skip the zero sigma

    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        sigma = sigmas[i]
        denoised = denoiser(x, sigma * s_in)
        d = to_d(x, sigma, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigma,
                    "denoised": denoised,
                }
            )
        dt = sigmas[i + 1] - sigma
        x = x + d * dt

    return x


@th.no_grad()
def iterative_colorization(
    distiller,
    images,
    x,
    ts,
    t_min=0.002,
    t_max=40.0,
    rho=7.0,
    steps=40,
    generator=None,
):
    def obtain_orthogonal_matrix():
        vector = np.asarray([0.2989, 0.5870, 0.1140])
        vector = vector / np.linalg.norm(vector)
        matrix = np.eye(3)
        matrix[:, 0] = vector
        matrix = np.linalg.qr(matrix)[0]
        if np.sum(matrix[:, 0]) < 0:
            matrix = -matrix
        return matrix

    Q = th.from_numpy(obtain_orthogonal_matrix()).to(dist_util.dev()).to(th.float32)
    mask = th.zeros(*x.shape[1:], device=dist_util.dev())
    mask[0, ...] = 1.0

    def replacement(x0, x1):
        x0 = th.einsum("bchw,cd->bdhw", x0, Q)
        x1 = th.einsum("bchw,cd->bdhw", x1, Q)

        x_mix = x0 * mask + x1 * (1.0 - mask)
        x_mix = th.einsum("bdhw,cd->bchw", x_mix, Q)
        return x_mix

    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])
    images = replacement(images, th.zeros_like(images))

    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        x0 = distiller(x, t * s_in)
        x0 = th.clamp(x0, -1.0, 1.0)
        x0 = replacement(images, x0)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        x = x0 + generator.randn_like(x) * np.sqrt(next_t**2 - t_min**2)

    return x, images


@th.no_grad()
def iterative_inpainting(
    distiller,
    images,
    x,
    ts,
    t_min=0.002,
    t_max=40.0,
    rho=7.0,
    steps=40,
    generator=None,
):
    from PIL import Image, ImageDraw, ImageFont

    image_size = x.shape[-1]

    # create a blank image with a white background
    img = Image.new("RGB", (image_size, image_size), color="white")

    # get a drawing context for the image
    draw = ImageDraw.Draw(img)

    # load a font
    font = ImageFont.truetype("arial.ttf", 250)

    # draw the letter "C" in black
    draw.text((50, 0), "S", font=font, fill=(0, 0, 0))

    # convert the image to a numpy array
    img_np = np.array(img)
    img_np = img_np.transpose(2, 0, 1)
    img_th = th.from_numpy(img_np).to(dist_util.dev())

    mask = th.zeros(*x.shape, device=dist_util.dev())
    mask = mask.reshape(-1, 7, 3, image_size, image_size)

    mask[::2, :, img_th > 0.5] = 1.0
    mask[1::2, :, img_th < 0.5] = 1.0
    mask = mask.reshape(-1, 3, image_size, image_size)

    def replacement(x0, x1):
        x_mix = x0 * mask + x1 * (1 - mask)
        return x_mix

    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])
    images = replacement(images, -th.ones_like(images))

    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        x0 = distiller(x, t * s_in)
        x0 = th.clamp(x0, -1.0, 1.0)
        x0 = replacement(images, x0)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        x = x0 + generator.randn_like(x) * np.sqrt(next_t**2 - t_min**2)

    return x, images


@th.no_grad()
def iterative_superres(
    distiller,
    images,
    x,
    ts,
    t_min=0.002,
    t_max=40.0,
    rho=7.0,
    steps=40,
    generator=None,
):
    patch_size = 8

    def obtain_orthogonal_matrix():
        vector = np.asarray([1] * patch_size**2)
        vector = vector / np.linalg.norm(vector)
        matrix = np.eye(patch_size**2)
        matrix[:, 0] = vector
        matrix = np.linalg.qr(matrix)[0]
        if np.sum(matrix[:, 0]) < 0:
            matrix = -matrix
        return matrix

    Q = th.from_numpy(obtain_orthogonal_matrix()).to(dist_util.dev()).to(th.float32)

    image_size = x.shape[-1]

    def replacement(x0, x1):
        x0_flatten = (
            x0.reshape(-1, 3, image_size, image_size)
            .reshape(
                -1,
                3,
                image_size // patch_size,
                patch_size,
                image_size // patch_size,
                patch_size,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(-1, 3, image_size**2 // patch_size**2, patch_size**2)
        )
        x1_flatten = (
            x1.reshape(-1, 3, image_size, image_size)
            .reshape(
                -1,
                3,
                image_size // patch_size,
                patch_size,
                image_size // patch_size,
                patch_size,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(-1, 3, image_size**2 // patch_size**2, patch_size**2)
        )
        x0 = th.einsum("bcnd,de->bcne", x0_flatten, Q)
        x1 = th.einsum("bcnd,de->bcne", x1_flatten, Q)
        x_mix = x0.new_zeros(x0.shape)
        x_mix[..., 0] = x0[..., 0]
        x_mix[..., 1:] = x1[..., 1:]
        x_mix = th.einsum("bcne,de->bcnd", x_mix, Q)
        x_mix = (
            x_mix.reshape(
                -1,
                3,
                image_size // patch_size,
                image_size // patch_size,
                patch_size,
                patch_size,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(-1, 3, image_size, image_size)
        )
        return x_mix

    def average_image_patches(x):
        x_flatten = (
            x.reshape(-1, 3, image_size, image_size)
            .reshape(
                -1,
                3,
                image_size // patch_size,
                patch_size,
                image_size // patch_size,
                patch_size,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(-1, 3, image_size**2 // patch_size**2, patch_size**2)
        )
        x_flatten[..., :] = x_flatten.mean(dim=-1, keepdim=True)
        return (
            x_flatten.reshape(
                -1,
                3,
                image_size // patch_size,
                image_size // patch_size,
                patch_size,
                patch_size,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(-1, 3, image_size, image_size)
        )

    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])
    images = average_image_patches(images)

    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        x0 = distiller(x, t * s_in)
        x0 = th.clamp(x0, -1.0, 1.0)
        x0 = replacement(images, x0)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        x = x0 + generator.randn_like(x) * np.sqrt(next_t**2 - t_min**2)

    return x, images


def compute_2nd_order_histogram_loss_fourier(density_in: th.Tensor, density_out: th.Tensor, target_ratio: float = 0.05) -> th.Tensor:
    """
    Compute 2nd order histogram loss using characteristic function (Fourier transform) approach.
    Optimized for MRI distributions with adaptive scaling.
    
    Args:
        density_in: Input density histogram [M]
        density_out: Output density histogram [M]
        target_ratio: Target ratio relative to 1st order KDE loss for adaptive scaling
        
    Returns:
        2nd order histogram loss based on characteristic function comparison
    """
    M = len(density_in)
    device = density_in.device
    
    # MRI-optimized frequency grid for characteristic function
    # MRI intensities typically range [0, 255] with specific tissue patterns
    t_max = 1.0  # Reduced for MRI stability (was 2.0)
    n_freqs = 20  # Increased resolution for better MRI tissue discrimination
    t_values = th.linspace(-t_max, t_max, n_freqs).to(device)
    
    # Assume uniform spacing in the histogram domain [0, 1] normalized
    dx = 1.0 / M
    x_values = th.arange(M, dtype=th.float32).to(device) * dx
    
    # Compute characteristic functions φ(t) = ∫ f(x) * e^(itx) dx
    char_func_in_list = []
    char_func_out_list = []
    
    for t_val in t_values:
        # Complex exponential: e^(itx) = cos(tx) + i*sin(tx)
        cos_factor = th.cos(t_val * x_values)
        sin_factor = th.sin(t_val * x_values)
        
        # Real and imaginary parts of characteristic functions
        char_real_in = th.sum(density_in * cos_factor) * dx
        char_imag_in = th.sum(density_in * sin_factor) * dx
        char_real_out = th.sum(density_out * cos_factor) * dx
        char_imag_out = th.sum(density_out * sin_factor) * dx
        
        # Store complex characteristic functions
        char_func_in = th.complex(char_real_in, char_imag_in)
        char_func_out = th.complex(char_real_out, char_imag_out)
        
        char_func_in_list.append(char_func_in)
        char_func_out_list.append(char_func_out)
    
    char_func_in_vals = th.stack(char_func_in_list)
    char_func_out_vals = th.stack(char_func_out_list)
    
    # Method 1: Direct computation of 2nd order moments (important for MRI contrast)
    # Compute E[X²] for both distributions - captures tissue intensity variance
    x_squared = x_values * x_values
    moment2_in = th.sum(density_in * x_squared) * dx
    moment2_out = th.sum(density_out * x_squared) * dx
    
    # Method 2: Use finite differences of characteristic function
    # φ''(t) ≈ (φ(t+h) - 2φ(t) + φ(t-h)) / h²
    h = t_values[1] - t_values[0] if len(t_values) > 1 else 0.1
    
    char_2nd_diff_list = []
    for i in range(1, len(t_values) - 1):
        char_2nd_diff = (char_func_in_vals[i+1] - 2*char_func_in_vals[i] + char_func_in_vals[i-1]) / (h*h)
        char_2nd_diff_list.append(char_2nd_diff)
    
    # if len(char_2nd_diff_list) > 0:
    #     char_2nd_in = th.stack(char_2nd_diff_list)
        
    #     char_2nd_diff_out_list = []
    #     for i in range(1, len(t_values) - 1):
    #         char_2nd_diff = (char_func_out_vals[i+1] - 2*char_func_out_vals[i] + char_func_out_vals[i-1]) / (h*h)
    #         char_2nd_diff_out_list.append(char_2nd_diff)
    #     char_2nd_out = th.stack(char_2nd_diff_out_list)
        
    #     # Compare 2nd order characteristic functions using real and imaginary parts
    #     char_diff = char_2nd_in - char_2nd_out
        
    #     # Separate real and imaginary parts for more stable loss computation
    #     real_diff = th.real(char_diff)
    #     imag_diff = th.imag(char_diff)
        
    #         # Compute losses on real and imaginary parts separately
    #     real_loss = th.mean(real_diff * real_diff)  # MSE on real part
    #     imag_loss = th.mean(imag_diff * imag_diff)  # MSE on imaginary part
        
    #     # Combine real and imaginary losses with equal weighting
    #     loss_2nd_char = real_loss + imag_loss
    # else:
    #     loss_2nd_char = th.tensor(0.0).to(device)


    if len(char_2nd_diff_list) > 0:
        char_2nd_in = th.stack(char_2nd_diff_list)
        
        char_2nd_diff_out_list = []
        for i in range(1, len(t_values) - 1):
            char_2nd_diff = (char_func_out_vals[i+1] - 2*char_func_out_vals[i] + char_func_out_vals[i-1]) / (h*h)
            char_2nd_diff_out_list.append(char_2nd_diff)
        char_2nd_out = th.stack(char_2nd_diff_out_list)
        
        # Compare 2nd order characteristic functions
        char_diff = char_2nd_in - char_2nd_out
        # Use magnitude of complex differences (robust for MRI noise)
        loss_2nd_char = th.mean(th.abs(char_diff))
        
        # Additional MRI-specific metric: phase coherence
        # Important for preserving edges and tissue boundaries
        phase_diff = th.angle(char_2nd_in) - th.angle(char_2nd_out)
        # Wrap phase differences to [-π, π]
        phase_diff = th.atan2(th.sin(phase_diff), th.cos(phase_diff))
        phase_loss = th.mean(th.abs(phase_diff))
        
        # Combine magnitude and phase losses
        loss_2nd_char = loss_2nd_char + 0.1 * phase_loss
    else:
        loss_2nd_char = th.tensor(0.0).to(device)
    
    # Combine both approaches with MRI-specific weighting
    # Moment difference is crucial for tissue contrast preservation
    moment_diff = th.abs(moment2_in - moment2_out)
    
    # Weighted combination optimized for MRI with adaptive scaling:
    base_loss = 0.4 * moment_diff + 0.6 * loss_2nd_char
    
    # Adaptive scaling based on target ratio
    # Estimate 1st order KDE loss magnitude for comparison
    kde_1st_estimate = th.sum(density_in * th.abs(th.log(density_in + 1e-8) - th.log(density_out + 1e-8)))
    
    if kde_1st_estimate.item() > 1e-6:
        # Scale to achieve target ratio with respect to 1st order loss
        adaptive_scaling = target_ratio * kde_1st_estimate / (base_loss + 1e-8)
        # Clamp scaling to reasonable bounds
        adaptive_scaling = th.clamp(adaptive_scaling, 1.0, 200.0)
    else:
        adaptive_scaling = 50.0  # Fallback scaling
    
    loss_2nd = adaptive_scaling * base_loss
    
    # Debug information
    print(f">>> DEBUG 2ND ORDER FOURIER (MRI): moment2_in={moment2_in.item():.8f}, moment2_out={moment2_out.item():.8f}", flush=True)
    print(f">>> DEBUG 2ND ORDER FOURIER (MRI): moment_diff={moment_diff.item():.8f}, char_loss={loss_2nd_char.item():.8f}", flush=True)
    print(f">>> DEBUG 2ND ORDER FOURIER (MRI): Final loss_2nd={loss_2nd.item():.8f}", flush=True)
    
    return loss_2nd


def compute_1st_order_histogram_loss(density_in: th.Tensor, density_out: th.Tensor) -> th.Tensor:
    """
    Compute 1st order histogram loss using KL divergence (KDE loss).
    
    Args:
        density_in: Input density histogram [M]
        density_out: Output density histogram [M]
        
    Returns:
        1st order histogram loss (KL divergence)
    """
    # KL divergence: D_KL(P||Q) = ∑ P(x) * log(P(x) / Q(x))
    kde_loss = th.sum(density_in * (th.log(density_in + 1e-8) - th.log(density_out + 1e-8)))
    return kde_loss
