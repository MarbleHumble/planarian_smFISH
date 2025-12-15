"""
GPU-native smFISH spot detection pipeline
Combines:
    1) LoG minima detection on GPU (with strength filtering)
    2) Raj lab-style plateau / elbow thresholding using 3D integrated intensity
    3) Optional 3D Gaussian fitting as a shape-validation step

Design philosophy (Raj 2008-consistent):
- LoG is a *proposal generator*, not a detector
- Integrated intensity separates signal from background
- Gaussian fitting validates morphology only after intensity filtering

Author: Elias Guan
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt

# ============================================================
# LoG / Gaussian utilities
# ============================================================

def gaussian_kernel_1d(sigma, device):
    radius = max(1, int(3 * sigma))
    x = torch.arange(-radius, radius + 1, device=device)
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def log_filter_gpu(image_np, sigma, device="cuda"):
    """Apply separable 3D Gaussian + Laplacian on GPU."""
    device = torch.device(device)
    x = torch.from_numpy(image_np).float().to(device)[None, None]

    sz, sy, sx = sigma
    kz = gaussian_kernel_1d(sz, device)[None, None, :, None, None]
    ky = gaussian_kernel_1d(sy, device)[None, None, None, :, None]
    kx = gaussian_kernel_1d(sx, device)[None, None, None, None, :]

    x = F.conv3d(x, kz, padding=(kz.shape[2] // 2, 0, 0))
    x = F.conv3d(x, ky, padding=(0, ky.shape[3] // 2, 0))
    x = F.conv3d(x, kx, padding=(0, 0, kx.shape[4] // 2))

    lap = (
        -6 * x
        + F.pad(x[:, :, 1:], (0, 0, 0, 0, 0, 1))
        + F.pad(x[:, :, :-1], (0, 0, 0, 0, 1, 0))
        + F.pad(x[:, :, :, 1:], (0, 0, 0, 1, 0, 0))
        + F.pad(x[:, :, :, :-1], (0, 0, 1, 0, 0, 0))
        + F.pad(x[:, :, :, :, 1:], (0, 1, 0, 0, 0, 0))
        + F.pad(x[:, :, :, :, :-1], (1, 0, 0, 0, 0, 0))
    )

    lap *= (sz ** 2 + sy ** 2 + sx ** 2)
    return lap.squeeze().cpu().numpy()


# ============================================================
# Local minima detection (with strength filtering)
# ============================================================

def local_minima_3d_strict(log_img, min_distance, depth_percentile=99.9, device="cuda"):
    """
    Detect local minima in LoG image and filter by response strength.
    This dramatically reduces background-induced minima.
    """
    dz, dy, dx = min_distance
    x = torch.from_numpy(log_img).to(device)[None, None]

    min_filt = -F.max_pool3d(
        -x,
        kernel_size=(2 * dz + 1, 2 * dy + 1, 2 * dx + 1),
        stride=1,
        padding=(dz, dy, dx),
    )

    depth_thresh = np.percentile(log_img, depth_percentile)
    mask = (x == min_filt) & (x < -depth_thresh)

    return mask.squeeze().nonzero(as_tuple=False).cpu().numpy().astype(np.int16)


# ============================================================
# Spot statistics (raw image domain)
# ============================================================

def spot_statistics(img, coords, radius):
    """
    Compute 3D integrated intensity and intensity-weighted radii
    for each candidate spot using the RAW image.
    """
    Z, Y, X = img.shape
    intensities = []
    radii = []

    for z, y, x in coords:
        z1, z2 = max(0, z - radius), min(Z, z + radius + 1)
        y1, y2 = max(0, y - radius), min(Y, y + radius + 1)
        x1, x2 = max(0, x - radius), min(X, x + radius + 1)
        sub = img[z1:z2, y1:y2, x1:x2]

        intensities.append(sub.sum())

        zz, yy, xx = np.indices(sub.shape)
        w = sub + 1e-6
        sz = np.sqrt(np.average((zz - zz.mean()) ** 2, weights=w))
        sy = np.sqrt(np.average((yy - yy.mean()) ** 2, weights=w))
        sx = np.sqrt(np.average((xx - xx.mean()) ** 2, weights=w))
        radii.append([sz, sy, sx])

    return np.asarray(intensities), np.asarray(radii)


# ============================================================
# Raj lab-style plateau / elbow detection
# ============================================================

def raj_plateau_threshold(intensities, smooth_window=11, slope_thresh=0.02,
                           min_fraction=0.05, save_plot=None):
    """
    Faithful Raj-style plateau detection based on slope stabilization
    of ranked 3D integrated intensities.
    """
    I = np.sort(intensities)[::-1]
    N = len(I)
    idx = np.arange(N)

    I_smooth = uniform_filter1d(I, smooth_window)
    dI = np.gradient(I_smooth)
    dI_norm = np.abs(dI / np.max(np.abs(dI)))

    plateau_mask = dI_norm < slope_thresh
    start_idx = int(min_fraction * N)
    valid = np.where(plateau_mask & (idx > start_idx))[0]

    if len(valid) == 0:
        elbow_idx = start_idx
    else:
        elbow_idx = valid[0]

    threshold = I[elbow_idx]

    if save_plot:
        plt.figure(figsize=(6, 4))
        plt.plot(I, label="Integrated intensity")
        plt.plot(I_smooth, label="Smoothed", linewidth=2)
        plt.axvline(elbow_idx, color="r", linestyle="--", label="Plateau onset")
        plt.xlabel("Spot rank")
        plt.ylabel("3D integrated intensity")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_plot)
        plt.close()

    return threshold, elbow_idx


# ============================================================
# Gaussian fitting (shape validation only)
# ============================================================

def gaussian_3d(coords, amp, z0, y0, x0, sz, sy, sx):
    z, y, x = coords
    return amp * np.exp(
        -(
            (z - z0) ** 2 / (2 * sz ** 2)
            + (y - y0) ** 2 / (2 * sy ** 2)
            + (x - x0) ** 2 / (2 * sx ** 2)
        )
    ).ravel()


def gaussian_fit_subset(
    img,
    coords,
    radius,
    expected_sigma,
    gaussian_fit_fraction=1.0,
    r2_threshold=0.8,
    seed=0,
):
    """Validate spot morphology using 3D Gaussian R²."""
    np.random.seed(seed)

    coords = np.asarray(coords)
    n_total = len(coords)

    if gaussian_fit_fraction < 1.0:
        n_sample = max(1, int(n_total * gaussian_fit_fraction))
        sample_idx = np.random.choice(n_total, n_sample, replace=False)
        coords_fit = coords[sample_idx]
    else:
        coords_fit = coords

    good_coords, bad_coords, r2_vals = [], [], []

    Z, Y, X = img.shape
    for z, y, x in coords_fit:
        z1, z2 = max(0, z - radius), min(Z, z + radius + 1)
        y1, y2 = max(0, y - radius), min(Y, y + radius + 1)
        x1, x2 = max(0, x - radius), min(X, x + radius + 1)
        sub = img[z1:z2, y1:y2, x1:x2]
        zz, yy, xx = np.indices(sub.shape)

        try:
            p0 = [sub.max(), *(np.array(sub.shape) // 2), *expected_sigma]
            popt, _ = curve_fit(
                gaussian_3d,
                (zz, yy, xx),
                sub.ravel(),
                p0=p0,
                maxfev=300,
            )
            fit = gaussian_3d((zz, yy, xx), *popt)
            residuals = sub.ravel() - fit
            r2 = 1 - np.sum(residuals**2) / np.sum(
                (sub.ravel() - sub.mean())**2
            )
            r2_vals.append(r2)

            if r2 >= r2_threshold:
                good_coords.append([z, y, x])
            else:
                bad_coords.append([z, y, x])
        except Exception:
            bad_coords.append([z, y, x])

    return (
        np.asarray(good_coords),
        np.asarray(bad_coords),
        np.asarray(r2_vals),
    )


# ============================================================
# Full GPU detection pipeline
# ============================================================

def detect_spots_gpu_full(
    image_np,
    sigma,
    min_distance,
    gaussian_radius=2,
    r2_threshold=0.8,
    random_seed=0,
    device="cuda",
    diagnostics_folder=None,
):
    """
    Full smFISH detection pipeline:
      1) LoG proposal generation
      2) Strict local minima + depth filtering
      3) Raj plateau intensity thresholding
      4) Gaussian shape validation
    """
    torch.manual_seed(random_seed)

    # --- LoG filtering ---
    log_img = -log_filter_gpu(image_np, sigma, device)

    # --- Local minima proposals ---
    coords = local_minima_3d_strict(log_img, min_distance, device=device)
    if len(coords) == 0:
        return np.empty((0, 3)), None, log_img, None, None, None, None

    # --- Spot statistics (RAW image) ---
    sum_intensities, radii = spot_statistics(image_np, coords, gaussian_radius)

    # --- Raj plateau threshold ---
    threshold, elbow_idx = raj_plateau_threshold(
        sum_intensities,
        save_plot=None if diagnostics_folder is None else f"{diagnostics_folder}/raj_plateau.png",
    )

    coords_int = coords[sum_intensities >= threshold]

    # --- Gaussian fitting ---
    good_coords, bad_coords, r2_vals = gaussian_fit_subset(
        image_np,
        coords_int,
        gaussian_radius,
        sigma,
        r2_threshold=r2_threshold,
        seed=random_seed,
    )

    # --- Diagnostics ---
    if diagnostics_folder is not None and len(r2_vals) > 0:
        plt.figure()
        plt.hist(r2_vals, bins=50)
        plt.axvline(r2_threshold, color="r")
        plt.title("Gaussian fit R² distribution")
        plt.savefig(f"{diagnostics_folder}/gaussian_r2.png")
        plt.close()

    return (
        good_coords,
        threshold,
        log_img,
        sum_intensities,
        radii,
        good_coords,
        bad_coords,
    )


# ============================================================
# Performance tuning
# ============================================================

def set_max_performance():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
