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

def mask_2d_gauss_accel(H, W, accel=8, sigma_frac=0.5):
    """
    2D Gaussian random sampling for ×8 acceleration.
    Produces low aliasing artifacts by weighting k-space center more heavily.
    """
    ky = np.linspace(-1, 1, H)[:, None]
    kx = np.linspace(-1, 1, W)[None, :]
    pdf = np.exp(-0.5 * ((ky / sigma_frac)**2 + (kx / sigma_frac)**2))
    pdf /= pdf.max()
    # scale so expected samples = H*W/accel
    scale = (H * W / accel) / pdf.sum()
    pdf_scaled = np.clip(pdf * scale, 0, 1)
    return (np.random.rand(H, W) < pdf_scaled).astype(np.float32)


def mask_1d_uniform_exact(H, W, accel=4, acs_frac=0.04):
    """
    1D uniform random sampling for ×4 acceleration,
    plus 4% central autocalibration (ACS) lines.
    """
    c = int(round(acs_frac * H))
    center = H // 2
    half = c // 2
    non_acs = np.ones(H, bool)
    non_acs[center-half : center-half + c] = False
    # solve p so that c + p*(H-c) = H/accel
    p = (H/accel - c) / non_acs.sum()
    mask = np.zeros((H, W), np.float32)
    mask[center-half : center-half + c, :] = 1.0
    rows = np.random.rand(H) < p
    for i in range(H):
        if non_acs[i] and rows[i]:
            mask[i, :] = 1.0
    return mask


def mask_1d_gauss_with_calib(H, W, accel=8, sigma_frac=0.5, acs_frac=0.04):
    """
    1D Gaussian-weighted sampling for ×8 acceleration,
    plus 4% ACS in the center for calibration.
    """
    c = int(round(acs_frac * H))
    center = H // 2
    half = c // 2
    non_acs = np.ones(H, bool)
    non_acs[center-half : center-half + c] = False
    ky = np.linspace(-1, 1, H)
    pdf = np.exp(-0.5 * (ky / sigma_frac)**2)
    pdf /= pdf.max()
    # scale so expected non-ACS = H/accel - c
    scale = (H/accel - c) / pdf[non_acs].sum()
    pdf_scaled = pdf * scale
    mask = np.zeros((H, W), np.float32)
    mask[center-half : center-half + c, :] = 1.0
    r = np.random.rand(H)
    for i in range(H):
        if non_acs[i] and r[i] < pdf_scaled[i]:
            mask[i, :] = 1.0
    return mask



def mask_variable_poisson(H, W, accel=15, pdf_exp=4, r_max=8, k=30):
    """
    Variable Poisson-disc sampling mask for k-space with acceleration constraint.
    Implements Algorithm 1 from the uploaded image.
    """
    cy, cx = H // 2, W // 2
    ys = np.arange(H) - cy
    xs = np.arange(W) - cx
    rr = np.sqrt(ys[:, None]**2 + xs[None, :]**2)

    # 1. Generate PDF for variable density
    pdf = np.exp(-0.5 * (rr / (max(H, W) / 2))**pdf_exp)
    pdf /= pdf.max()

    # 2. Scale so expected number of samples = total / accel
    scale = (H * W / accel) / pdf.sum()
    pdf_scaled = np.clip(pdf * scale, 0, 1)

    # 3. Define r(x, y) = r_max * (1 - pdf_scaled)
    r_map = r_max * (1 - pdf_scaled)
    r_map = np.clip(r_map, r_max * 0.3, r_max)  # Avoid extremely small radius

    # 4. Setup background grid
    cell_size = r_max / np.sqrt(2)
    grid_H, grid_W = int(np.ceil(H / cell_size)), int(np.ceil(W / cell_size))
    grid = [[[] for _ in range(grid_W)] for _ in range(grid_H)]

    def get_cell_coords(y, x):
        return int(y // cell_size), int(x // cell_size)

    def in_bounds(y, x):
        return 0 <= y < H and 0 <= x < W

    def is_far_enough(y, x, r_val):
        gi, gj = get_cell_coords(y, x)
        for ii in range(max(0, gi - 2), min(grid_H, gi + 3)):
            for jj in range(max(0, gj - 2), min(grid_W, gj + 3)):
                for py, px in grid[ii][jj]:
                    if (y - py)**2 + (x - px)**2 < r_val**2:
                        return False
        return True

    # 5. Initialize with a random point
    while True:
        y0 = np.random.randint(H)
        x0 = np.random.randint(W)
        if np.random.rand() < pdf_scaled[y0, x0]:
            break

    points = [(y0, x0)]
    active = [(y0, x0)]
    gi, gj = get_cell_coords(y0, x0)
    grid[gi][gj].append((y0, x0))

    # 6. Main loop
    while active:
        idx = np.random.randint(len(active))
        yi, xi = active[idx]
        ri = r_map[yi, xi]
        found = False

        for _ in range(k):
            theta = 2 * np.pi * np.random.rand()
            rad = ri * (1 + np.random.rand())  # radius in [r, 2r]
            yj = int(round(yi + rad * np.sin(theta)))
            xj = int(round(xi + rad * np.cos(theta)))
            if not in_bounds(yj, xj):
                continue
            rj = r_map[yj, xj]
            if not is_far_enough(yj, xj, rj):
                continue
            points.append((yj, xj))
            active.append((yj, xj))
            gi, gj = get_cell_coords(yj, xj)
            grid[gi][gj].append((yj, xj))
            found = True
            break

        if not found:
            active.pop(idx)

    # 7. Create final binary mask
    mask = np.zeros((H, W), dtype=np.float32)
    for y, x in points:
        mask[y, x] = 1.0

    return mask

# ——— zero-filled IFFT reconstruction ———
def recon_from_mask(img, mask):
    k = np.fft.fftshift(np.fft.fft2(img))
    rec = np.abs(np.fft.ifft2(np.fft.ifftshift(k * mask)))
    rec = (rec - rec.min()) / (rec.max() - rec.min() + 1e-5)
    return rec

# ——— main: save everything ———
if __name__ == "__main__":
    series = "/radraid/kzhao/fastmri_prostate_DICOMS_IDS_001_312/DICOMS/001/AX_DIFFUSION_ADC"
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)

    img = load_one_slice(series)
    H, W = img.shape

    methods = {
        "2D_Gauss_8x":      mask_2d_gauss_accel(H, W, accel=8,  sigma_frac=0.3),
        "1D_Uniform_4x":    mask_1d_uniform_exact(H, W, accel=4,  acs_frac=0.04),
        "1D_Gauss_8x_ACS":  mask_1d_gauss_with_calib(H, W, accel=8,  sigma_frac=0.3, acs_frac=0.04),
        # 4. ×15 VD Poisson: most aggressive undersampling; state-of-the-art, minimal artifacts given same budget (Dwork et al., 2021)
        "VD_Poisson_15x":  mask_variable_poisson(H, W, accel=15, pdf_exp=4, r_max=4, k=30),
    }

    Image.fromarray((img * 255).astype(np.uint8)).save(os.path.join(out_dir, "original.png"))
    for name, mask in methods.items():
        Image.fromarray((mask * 255).astype(np.uint8)).save(os.path.join(out_dir, f"mask_{name}.png"))
        rec = recon_from_mask(img, mask)
        Image.fromarray((rec * 255).astype(np.uint8)).save(os.path.join(out_dir, f"recon_{name}.png"))
        print(f"Saved mask_{name}.png and recon_{name}.png")
