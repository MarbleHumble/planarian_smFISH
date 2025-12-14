"""
GPU-native smFISH spot detection (Big-FISH inspired, optimized)
Author: Elias Guan
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from typing import Tuple


# ============================================================
# Gaussian / LoG utilities
# ============================================================
def gaussian_kernel_1d(sigma, device):
    radius = max(1, int(3 * sigma))
    x = torch.arange(-radius, radius + 1, device=device)
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def log_filter_gpu(image_np, sigma, device="cuda"):
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

    lap *= sz**2 + sy**2 + sx**2
    return lap.squeeze().cpu().numpy()


# ============================================================
# Local minima detection
# ============================================================
def local_minima_3d(log_img, min_distance):
    dz, dy, dx = min_distance
    x = log_img[None, None]
    min_filt = -F.max_pool3d(
        -x,
        kernel_size=(2 * dz + 1, 2 * dy + 1, 2 * dx + 1),
        stride=1,
        padding=(dz, dy, dx),
    )
    mask = (x == min_filt) & (x < 0)
    return mask.squeeze().nonzero(as_tuple=False)


# ============================================================
# Fast per-spot statistics (ALL spots)
# ============================================================
def spot_statistics(img, coords, radius):
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

    return np.array(intensities), np.array(radii)


# ============================================================
# Gaussian fitting (SUBSET only)
# ============================================================
def gaussian_3d(coords, amp, z0, y0, x0, sz, sy, sx):
    z, y, x = coords
    return (
        amp
        * np.exp(
            -(
                (z - z0) ** 2 / (2 * sz**2)
                + (y - y0) ** 2 / (2 * sy**2)
                + (x - x0) ** 2 / (2 * sx**2)
            )
        )
    ).ravel()


def gaussian_fit_subset(
    img,
    coords,
    radius,
    expected_sigma,
    r2_threshold,
    fit_fraction,
    seed,
):
    np.random.seed(seed)
    n = len(coords)
    n_fit = max(50, int(fit_fraction * n))
    idx = np.random.choice(n, n_fit, replace=False)

    good_intensities = []
    good_coords = []
    bad_coords = []

    Z, Y, X = img.shape

    for i in idx:
        z, y, x = coords[i]
        z1, z2 = max(0, z - radius), min(Z, z + radius + 1)
        y1, y2 = max(0, y - radius), min(Y, y + radius + 1)
        x1, x2 = max(0, x - radius), min(X, x + radius + 1)

        sub = img[z1:z2, y1:y2, x1:x2]
        zz, yy, xx = np.indices(sub.shape)

        try:
            p0 = [sub.max(), *np.array(sub.shape) // 2, *expected_sigma]
            popt, _ = curve_fit(
                gaussian_3d,
                (zz, yy, xx),
                sub.ravel(),
                p0=p0,
                maxfev=200,
            )
            residuals = sub.ravel() - gaussian_3d((zz, yy, xx), *popt)
            r2 = 1 - np.sum(residuals**2) / np.sum(
                (sub.ravel() - sub.mean()) ** 2
            )
            if r2 >= r2_threshold:
                good_intensities.append(sub.sum())
                good_coords.append([z, y, x])
            else:
                bad_coords.append([z, y, x])
        except Exception:
            bad_coords.append([z, y, x])

    return (
        np.array(good_intensities),
        np.array(good_coords),
        np.array(bad_coords),
    )


# ============================================================
# Full detector
# ============================================================
def detect_spots_gpu(
    image_np,
    sigma,
    min_distance,
    threshold=None,
    gaussian_radius=2,
    spots_radius_detection=True,
    gaussian_fit_fraction=0.1,
    intensity_percentile_cutoff=99,
    r2_threshold=0.8,
    random_seed=0,
    device="cuda",
):
    """
    GPU-native smFISH spot detection (memory-safe for large 3D images).

    Returns:
        coords (np.ndarray): final spot coordinates
        log_img (np.ndarray): LoG filtered image
        intensities (np.ndarray): spot intensities
        radii (np.ndarray): spot radii (σz,σy,σx)
        good_c (np.ndarray): coordinates used in Gaussian fit subset
        bad_c (np.ndarray): coordinates not fitting Gaussian
    """

    import torch
    import numpy as np

    # -------------------------
    # Compute LoG on GPU
    # -------------------------
    log_img = -log_filter_gpu(image_np, sigma, device)
    log_t = torch.from_numpy(log_img).to(device)

    # -------------------------
    # Automatic threshold if None (memory-safe)
    # -------------------------
    if threshold is None:
        # Move negative LoG values to CPU
        neg_vals = log_t[log_t < 0].cpu()
        if neg_vals.numel() == 0:
            raise RuntimeError("No negative LoG values found for automatic thresholding.")

        # Subsample if too many values
        max_samples = int(1e6)
        if neg_vals.numel() > max_samples:
            idx = torch.randperm(neg_vals.numel())[:max_samples]
            neg_vals = neg_vals[idx]

        # Compute quantile for threshold
        threshold = torch.quantile(neg_vals, 0.999).item()

    # -------------------------
    # Thresholding
    # -------------------------
    log_t = torch.where(log_t <= threshold, log_t, torch.zeros_like(log_t))

    # -------------------------
    # Local minima detection
    # -------------------------
    coords = local_minima_3d(log_t, min_distance).cpu().numpy()

    # -------------------------
    # Spot statistics (intensity & radii)
    # -------------------------
    intensities, radii = spot_statistics(
        image_np, coords, gaussian_radius
    )

    # -------------------------
    # Gaussian fit on subset
    # -------------------------
    good_c, bad_c = None, None
    if gaussian_fit_fraction > 0 and len(coords) > 0:
        good_I, good_c, bad_c = gaussian_fit_subset(
            image_np,
            coords,
            gaussian_radius,
            sigma,
            r2_threshold,
            gaussian_fit_fraction,
            random_seed,
        )
        # Filter based on intensity percentile of good fits
        I_cut = np.percentile(good_I, intensity_percentile_cutoff)
        keep = intensities >= I_cut
        coords = coords[keep]
        intensities = intensities[keep]
        radii = radii[keep]

    return coords, log_img, intensities, radii, good_c, bad_c


# ============================================================
# Visualization
# ============================================================
def plot_spot_example(img, coord, radius, save_path, title):
    z, y, x = coord
    Z, Y, X = img.shape

    z1, z2 = max(0, z - radius), min(Z, z + radius + 1)
    y1, y2 = max(0, y - radius), min(Y, y + radius + 1)
    x1, x2 = max(0, x - radius), min(X, x + radius + 1)

    sub = img[z1:z2, y1:y2, x1:x2]

    plt.figure(figsize=(9, 3))
    plt.suptitle(title)

    plt.subplot(1, 3, 1)
    plt.imshow(sub[sub.shape[0] // 2], cmap="hot")
    plt.title("XY")

    plt.subplot(1, 3, 2)
    plt.imshow(sub[:, sub.shape[1] // 2, :], cmap="hot")
    plt.title("XZ")

    plt.subplot(1, 3, 3)
    plt.imshow(sub[:, :, sub.shape[2] // 2], cmap="hot")
    plt.title("YZ")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ============================================================
# Performance
# ============================================================
def set_max_performance():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")