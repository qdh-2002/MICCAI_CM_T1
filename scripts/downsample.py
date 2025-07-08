import os
import numpy as np
import pydicom
from PIL import Image

# ——— load one DICOM slice ———
def load_one_slice(path_to_series):
    fn = sorted(f for f in os.listdir(path_to_series) if f.endswith('.dcm'))[0]
    d = pydicom.dcmread(os.path.join(path_to_series, fn))
    img = d.pixel_array.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-5)
    return img

# ——— mask generators ———
def mask_2d_gauss(H, W, sigma_frac=0.5):
    ky = np.linspace(-1,1,H)[:,None]
    kx = np.linspace(-1,1,W)[None,:]
    pdf = np.exp(-0.5*((ky/sigma_frac)**2 + (kx/sigma_frac)**2))
    pdf /= pdf.max()
    return (np.random.rand(H,W) < pdf).astype(np.float32)

def mask_1d_uniform(H, W, accel=4):
    p = 1/accel
    lines = (np.random.rand(H) < p)
    m = np.zeros((H,W),float); m[lines,:]=1
    return m

def mask_1d_gauss(H, W, sigma_frac=0.5):
    ky = np.linspace(-1,1,H)
    pdf = np.exp(-0.5*(ky/sigma_frac)**2)
    pdf /= pdf.max()
    lines = (np.random.rand(H) < pdf)
    m = np.zeros((H,W),float); m[lines,:]=1
    return m



def mask_vd_poisson(H, W, r_min=8, pdf_exp=4):
    """
    Variable-density Poisson-disk on a H×W grid:
     1) Compute radial PDF: p(r) ∝ exp(–0.5*(r/kmax)**pdf_exp)
     2) Thinning: retain each (i,j) with probability p
     3) Poisson-disk: scan shuffled survivors and accept if ≥r_min from all previous
    """
    # 1) radial PDF
    cy, cx = H//2, W//2
    ys = np.arange(H) - cy
    xs = np.arange(W) - cx
    rr = np.sqrt(ys[:,None]**2 + xs[None,:]**2)
    pdf = np.exp(-0.5 * (rr / (max(H,W)/2))**pdf_exp)
    pdf /= pdf.max()

    # 2) thinning
    coords = [(i,j) for i in range(H) for j in range(W)
              if np.random.rand() < pdf[i,j]]
    np.random.shuffle(coords)

    # 3) Poisson-disk filtering
    accepted = []
    mask = np.zeros((H,W), np.float32)
    for (i,j) in coords:
        # if this point is at least r_min from all accepted, keep it
        too_close = False
        for (ai,aj) in accepted:
            if (i - ai)**2 + (j - aj)**2 < r_min**2:
                too_close = True
                break
        if not too_close:
            accepted.append((i,j))
            mask[i,j] = 1.0

    return mask


# ——— zero-filled IFFT reconstruction ———
def recon_from_mask(img, mask):
    k = np.fft.fftshift(np.fft.fft2(img))
    rec = np.abs(np.fft.ifft2(np.fft.ifftshift(k*mask)))
    # normalize to [0,1]
    rec = (rec - rec.min()) / (rec.max()-rec.min()+1e-5)
    return rec

# … keep your existing imports and load_one_slice, mask_2d_gauss, mask_1d_gauss, recon_from_mask, etc.

def mask_1d_uniform_with_calib(H, W, accel=4, calib_lines=16):
    """
    1D uniform random lines + fully sample `calib_lines` around center.
    """
    mask = np.zeros((H, W), np.float32)
    # 1) calibration region
    center = H // 2
    half = calib_lines // 2
    mask[center-half : center+half, :] = 1.0

    # 2) uniform sampling outside calib
    p = 1 / accel
    for i in range(H):
        if mask[i,0] == 0 and np.random.rand() < p:
            mask[i, :] = 1.0

    return mask

def mask_poisson_disk_circle(H, W, r_min=8, circle_radius=None, k=30):
    """
    Poisson-disk sampling *inside* a circle, on a H×W grid.

    Args:
      H, W            : grid size
      r_min           : minimum distance between samples (in pixels)
      circle_radius   : radius of the inclusion circle (in pixels). 
                        Defaults to min(H,W)/2 (full inscribed circle).
      k               : Bridson’s “rejection” parameter
    
    Returns:
      mask (H×W float32): 1 where a sample was placed, 0 elsewhere
    """
    if circle_radius is None:
        circle_radius = min(H, W) / 2
    cy, cx = H/2, W/2

    # Helper: is a point in the inclusion circle?
    def in_circle(pt):
        y, x = pt
        return (y - cy)**2 + (x - cx)**2 <= circle_radius**2

    # Bridson’s algorithm in continuous coords, restricted to circle
    cell = r_min / np.sqrt(2)
    grid_shape = (int(np.ceil(H / cell)), int(np.ceil(W / cell)))
    grid = -np.ones(grid_shape, dtype=int)
    samples = []
    active = []

    # 1) first sample: random in circle
    while True:
        y0 = np.random.rand() * H
        x0 = np.random.rand() * W
        if in_circle((y0, x0)):
            samples.append((y0, x0))
            gy, gx = int(y0 // cell), int(x0 // cell)
            grid[gy, gx] = 0
            active.append(0)
            break

    # 2) generate points
    while active:
        idx = np.random.choice(active)
        sy, sx = samples[idx]
        found = False
        for _ in range(k):
            theta = 2 * np.pi * np.random.rand()
            rad = r_min * (1 + np.random.rand())
            ny, nx = sy + rad * np.sin(theta), sx + rad * np.cos(theta)
            if not (0 <= ny < H and 0 <= nx < W): 
                continue
            if not in_circle((ny, nx)):
                continue
            gy, gx = int(ny // cell), int(nx // cell)
            y0 = max(0, gy - 2)
            y1 = min(grid_shape[0], gy + 3)
            x0 = max(0, gx - 2)
            x1 = min(grid_shape[1], gx + 3)
            ok = True
            for ui in range(y0, y1):
                for uj in range(x0, x1):
                    sidx = grid[ui, uj]
                    if sidx >= 0:
                        py, px = samples[sidx]
                        if (py - ny)**2 + (px - nx)**2 < r_min**2:
                            ok = False
                            break
                if not ok:
                    break
            if ok:
                samples.append((ny, nx))
                grid[gy, gx] = len(samples) - 1
                active.append(len(samples) - 1)
                found = True
                break
        if not found:
            active.remove(idx)

    # 3) rasterize to integer grid
    mask = np.zeros((H, W), dtype=np.float32)
    for (y, x) in samples:
        iy, ix = int(round(y)), int(round(x))
        mask[iy, ix] = 1.0

    return mask

import numpy as np

def mask_poisson_disk_circle_with_calib(
    H, W,
    r_min=8,
    circle_radius=None,
    calib_radius=16,
    k=30
):
    """
    Poisson-disk sampling inside a circle of radius `circle_radius`,
    plus a fully-sampled central disk of radius `calib_radius`.
    """
    if circle_radius is None:
        circle_radius = min(H, W) / 2
    cy, cx = H/2, W/2

    def in_circle(y, x, rad):
        return (y - cy)**2 + (x - cx)**2 <= rad**2

    # Bridson’s Poisson-disk
    cell = r_min / np.sqrt(2)
    grid_shape = (int(np.ceil(H / cell)), int(np.ceil(W / cell)))
    grid = -np.ones(grid_shape, dtype=int)
    samples = []
    active = []

    # first sample
    while True:
        y0 = np.random.rand()*H
        x0 = np.random.rand()*W
        if in_circle(y0, x0, circle_radius):
            samples.append((y0, x0))
            gy, gx = int(y0//cell), int(x0//cell)
            grid[gy, gx] = 0
            active.append(0)
            break

    # generate others
    while active:
        idx = np.random.choice(active)
        sy, sx = samples[idx]
        found = False
        for _ in range(k):
            theta = 2*np.pi*np.random.rand()
            rad = r_min*(1 + np.random.rand())
            ny = sy + rad*np.sin(theta)
            nx = sx + rad*np.cos(theta)
            if not (0 <= ny < H and 0 <= nx < W): 
                continue
            if not in_circle(ny, nx, circle_radius):
                continue
            gy, gx = int(ny//cell), int(nx//cell)
            y0 = max(0, gy-2)
            y1 = min(grid_shape[0], gy+3)
            x0 = max(0, gx-2)
            x1 = min(grid_shape[1], gx+3)
            ok = True
            for ui in range(y0, y1):
                for uj in range(x0, x1):
                    sidx = grid[ui, uj]
                    if sidx >= 0:
                        py, px = samples[sidx]
                        if (py-ny)**2 + (px-nx)**2 < r_min**2:
                            ok = False
                            break
                if not ok:
                    break
            if ok:
                samples.append((ny, nx))
                grid[gy, gx] = len(samples)-1
                active.append(len(samples)-1)
                found = True
                break
        if not found:
            active.remove(idx)

    # rasterize with clipping
    mask = np.zeros((H, W), dtype=np.float32)
    for (y, x) in samples:
        iy = int(round(y))
        ix = int(round(x))
        # clip into valid range
        iy = max(0, min(H-1, iy))
        ix = max(0, min(W-1, ix))
        mask[iy, ix] = 1.0

    # force-include central calib disk
    ys = np.arange(H) - cy
    xs = np.arange(W) - cx
    rr = np.sqrt(ys[:,None]**2 + xs[None,:]**2)
    mask[rr <= calib_radius] = 1.0

    return mask

# ——— main: save everything ———
if __name__ == "__main__":
    series = "/radraid/kzhao/fastmri_prostate_DICOMS_IDS_001_312/DICOMS/001/AX_DIFFUSION_ADC"
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)

    img = load_one_slice(series)
    H, W = img.shape

    methods = {
        "2D_Gauss":     mask_2d_gauss(H,W, sigma_frac=0.3),
        "1D_Uniform":   mask_1d_uniform_with_calib(H,W, accel=4, calib_lines=9),
        "1D_Gauss":     mask_1d_gauss(H,W, sigma_frac=0.4),
        "VD_Poisson":  mask_poisson_disk_circle_with_calib(
        H, W,
        r_min=4,            # Poisson min distance
        circle_radius=120,  # overall support circle
        calib_radius=12,    # fully-sampled center disk
        k=30
    ),
    }

    # save original
    Image.fromarray((img*255).astype(np.uint8)).save(os.path.join(out_dir, "original.png"))

    for name, mask in methods.items():
        # save mask (as 0/255 image)
        mask_img = (mask*255).astype(np.uint8)
        Image.fromarray(mask_img).save(os.path.join(out_dir, f"mask_{name}.png"))

        # reconstruct & save
        rec = recon_from_mask(img, mask)
        rec_img = (rec*255).astype(np.uint8)
        Image.fromarray(rec_img).save(os.path.join(out_dir, f"recon_{name}.png"))

        print(f"Saved mask_{name}.png and recon_{name}.png")
