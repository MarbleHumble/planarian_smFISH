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
    intensity_percentile_cutoff=99.0,
    r2_threshold=0.8,
    random_seed=0,
    device="cuda",
):
    """
    GPU-native smFISH spot detection using LoG minima.

    Returns
    -------
    coords_all : (N, 3) np.ndarray
        Spot centroids (z, y, x)
    log_img : np.ndarray
        LoG-filtered image (NumPy)
    sum_intensities : np.ndarray
        Integrated intensity per spot (or empty)
    radii : np.ndarray
        Estimated spot radii (or empty)
    good_coords : np.ndarray
        Coordinates passing Gaussian fit (or None)
    bad_coords : np.ndarray
        Coordinates failing Gaussian fit (or None)
    """

    import torch
    import numpy as np

    torch.manual_seed(random_seed)

    # -------------------------
    # Defaults (safe return)
    # -------------------------
    coords_all = np.empty((0, 3), dtype=np.int16)
    sum_intensities = np.array([])
    radii = np.array([])
    good_coords = None
    bad_coords = None

    # -------------------------
    # Compute LoG on GPU
    # -------------------------
    log_img = -log_filter_gpu(image_np, sigma, device)
    log_t = torch.from_numpy(log_img).to(device)

    # -------------------------
    # Automatic threshold (LoG space)
    # -------------------------
    if threshold is None:
        neg_vals = log_t[log_t < 0]

        if neg_vals.numel() == 0:
            # No signal at all → return safely
            return (
                coords_all,
                log_img,
                sum_intensities,
                radii,
                good_coords,
                bad_coords,
            )

        # Subsample for memory safety
        if neg_vals.numel() > 1_000_000:
            idx = torch.randperm(neg_vals.numel(), device=device)[:1_000_000]
            neg_vals = neg_vals[idx]

        threshold = torch.quantile(neg_vals, 0.999).item()

    # -------------------------
    # Thresholding
    # -------------------------
    log_t = torch.where(log_t <= threshold, log_t, torch.zeros_like(log_t))

    # -------------------------
    # Local minima detection
    # -------------------------
    coords_t = local_minima_3d(log_t, min_distance)

    if coords_t.numel() == 0:
        return (
            coords_all,
            log_img,
            sum_intensities,
            radii,
            good_coords,
            bad_coords,
        )

    coords_all = coords_t.cpu().numpy().astype(np.int16)

    # -------------------------
    # Spot intensity & radius estimation
    # -------------------------
    if spots_radius_detection:
        sum_intensities, radii = spot_statistics(
            image_np,
            coords_all,
            gaussian_radius,
        )

    # -------------------------
    # Optional Gaussian fit (subset)
    # -------------------------
    if (
        gaussian_fit_fraction > 0
        and sum_intensities is not None
        and len(coords_all) > 0
    ):
        good_I, good_coords, bad_coords = gaussian_fit_subset(
            image_np,
            coords_all,
            gaussian_radius,
            sigma,
            r2_threshold,
            gaussian_fit_fraction,
            random_seed,
        )

        # Intensity cutoff (for logging / QC)
        if len(good_I) > 0:
            _ = np.percentile(good_I, intensity_percentile_cutoff)

    # -------------------------
    # Final return
    # -------------------------
    return (
        coords_all,
        log_img,
        sum_intensities,
        radii,
        good_coords,
        bad_coords,
    )


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