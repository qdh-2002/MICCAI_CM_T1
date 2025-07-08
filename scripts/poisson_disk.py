import random
import math
import time
from PIL import Image

# ------------------------------------------------------------------
# 1) Radius function factory: r(p) = ( ‖p‖ + δ) / γ
# ------------------------------------------------------------------
def make_r_func(gamma, delta=0.15):
    return lambda p: (math.hypot(p[0], p[1]) + delta) / gamma

# ------------------------------------------------------------------
# 2) Fast Poisson-disc sampler (Tulleken / Dwork style)
# ------------------------------------------------------------------
class FastPoissonSampler:
    def __init__(self, width, height, r_cell, r_func, k=30):
        self.W, self.H = width, height
        self.r_cell, self.r_func, self.k = r_cell, r_func, k
        self.cell_size = r_cell / math.sqrt(2)
        self.grid_w = math.ceil(self.W  / self.cell_size)
        self.grid_h = math.ceil(self.H  / self.cell_size)
        self.bg_grid = [[[] for _ in range(self.grid_w)] for _ in range(self.grid_h)]
        self.points = []
        self.active = []

    def _cell_coords(self, p):
        return int(p[0] // self.cell_size), int(p[1] // self.cell_size)

    def sample(self):
        # seed
        init = (random.uniform(0, self.W), random.uniform(0, self.H))
        self.points.append(init)
        self.active.append(0)
        cx, cy = self._cell_coords(init)
        self.bg_grid[cy][cx].append(0)

        # main loop
        while self.active:
            idx = random.choice(self.active)
            base = self.points[idx]
            r_b = self.r_func(base)
            found = False
            for _ in range(self.k):
                rho   = random.uniform(r_b, 2*r_b)
                theta = random.uniform(0, 2*math.pi)
                x_new = base[0] + rho * math.cos(theta)
                y_new = base[1] + rho * math.sin(theta)
                if not (0 <= x_new < self.W and 0 <= y_new < self.H):
                    continue
                p = (x_new, y_new)
                cx2, cy2 = self._cell_coords(p)
                too_close = False
                for oi in self.bg_grid[cy2][cx2]:
                    q = self.points[oi]
                    if math.hypot(q[0]-p[0], q[1]-p[1]) < self.r_func(p):
                        too_close = True
                        break
                if too_close:
                    continue
                # accept
                new_idx = len(self.points)
                self.points.append(p)
                self.active.append(new_idx)
                self.bg_grid[cy2][cx2].append(new_idx)
                found = True
            if not found:
                self.active.remove(idx)

        return self.points

# ------------------------------------------------------------------
# 3) Helper to draw a mask from a point set
# ------------------------------------------------------------------
def points_to_mask(points, bounds, sizeX, sizeY):
    dx = (bounds[2] - bounds[0]) / sizeX
    dy = (bounds[3] - bounds[1]) / sizeY
    mask = Image.new("L", (sizeX, sizeY), 0)
    pix  = mask.load()
    for x,y in points:
        ix = int((x - bounds[0]) / dx)
        iy = int((y - bounds[1]) / dy)
        if 0 <= ix < sizeX and 0 <= iy < sizeY:
            pix[ix,iy] = 255
    n_samples = sum(pix[x,y] == 255 for x in range(sizeX) for y in range(sizeY))
    return mask, n_samples

# ------------------------------------------------------------------
# 4) Binary-search helper
# ------------------------------------------------------------------
def find_gamma(acc_func, lo, hi, target, tol=0.01, max_iter=100):
    print(f"Finding γ for target_accel={target:.2f}…")
    for i in range(max_iter):
        mid = 0.5*(lo + hi)
        diff, acc = acc_func(mid)
        if abs(diff) < tol:
            return mid, acc, i+1
        # accel(γ) decreases as γ increases
        if diff > 0:
            lo = mid
        else:
            hi = mid
    return mid, acc, max_iter

# ------------------------------------------------------------------
# 5) Main experiment
# ------------------------------------------------------------------
if __name__ == "__main__":
    sizeX = sizeY = 64
    bounds = [-0.5, -0.5, 0.5, 0.5]
    k = 30
    target_accel = 4.5

    print(f"Running Poisson disk sampling with {sizeX}×{sizeY}, target accel={target_accel}×")

    # initial γ and radii
    run_gamma = 150
    r_func    = make_r_func(run_gamma)
    min_r     = r_func((0,0))
    max_r     = r_func((bounds[2], bounds[3]))
    print(f"Initial γ={run_gamma:.3f}, min_r={min_r:.3f}, max_r={max_r:.3f}")

    # --- initial masks ---
    for name, r_cell in [("Tulleken", max_r), ("Dwork", min_r)]:
        sampler = FastPoissonSampler(sizeX, sizeY, r_cell, r_func, k)
        pts     = sampler.sample()
        mask, _ = points_to_mask(pts, bounds, sizeX, sizeY)
        mask.save(f"{name.lower()}_initial.png")
        print(f"{name} initial: points={len(pts)}, accel={(sizeX*sizeY)/len(pts):.2f}×")

    # define the two accel-difference functions now that sizeX, etc. exist
    def acc_diff_dwork(gamma):
        rf   = make_r_func(gamma)
        rmin = rf((0,0))
        pts  = FastPoissonSampler(sizeX, sizeY, rmin, rf, k).sample()
        acc  = sizeX*sizeY / len(pts)
        return acc - target_accel, acc

    def acc_diff_tulleken(gamma):
        rf   = make_r_func(gamma)
        rmax = rf((bounds[2], bounds[3]))
        pts  = FastPoissonSampler(sizeX, sizeY, rmax, rf, k).sample()
        acc  = sizeX*sizeY / len(pts)
        return acc - target_accel, acc

    # binary-search for γ
    d_gamma, d_acc, d_it = find_gamma(acc_diff_dwork,   min_r*sizeX, max_r*sizeX, target_accel)
    t_gamma, t_acc, t_it = find_gamma(acc_diff_tulleken, min_r*sizeX, max_r*sizeX, target_accel)
    print(f"Dwork γ → {d_gamma:.3f} (acc={d_acc:.3f}× in {d_it} iters)")
    print(f"Tulleken γ → {t_gamma:.3f} (acc={t_acc:.3f}× in {t_it} iters)")

    # save final masks
    for name, gamma in [("dwork", d_gamma), ("tulleken", t_gamma)]:
        rf     = make_r_func(gamma)
        r_cell = rf((0,0)) if name=="dwork" else rf((bounds[2],bounds[3]))
        pts    = FastPoissonSampler(sizeX, sizeY, r_cell, rf, k).sample()
        mask, _ = points_to_mask(pts, bounds, sizeX, sizeY)
        mask.save(f"{name}_gamma.png")

    print("Saved: tulleken_initial.png, dwork_initial.png, tulleken_gamma.png, dwork_gamma.png")
