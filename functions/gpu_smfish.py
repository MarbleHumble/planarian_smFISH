"""
GPU-native smFISH spot detection pipeline
Combines:
    1) LoG minima detection on GPU
    2) Raj lab-style plateau/elbow thresholding using 3D spot intensity
    3) Optional Gaussian fitting to filter spots
Author: Elias Guan
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# ============================================================
# LoG / Gaussian utilities
# ============================================================
def gaussian_kernel_1d(sigma, device):
    radius = max(1, int(3 * sigma))
    x = torch.arange(-radius, radius + 1, device=device)
    k = torch.exp(-(x**2) / (2*sigma**2))
    return k / k.sum()


def log_filter_gpu(image_np, sigma, device="cuda"):
    device = torch.device(device)
    x = torch.from_numpy(image_np).float().to(device)[None, None]

    sz, sy, sx = sigma
    kz = gaussian_kernel_1d(sz, device)[None, None, :, None, None]
    ky = gaussian_kernel_1d(sy, device)[None, None, None, :, None]
    kx = gaussian_kernel_1d(sx, device)[None, None, None, None, :]

    x = F.conv3d(x, kz, padding=(kz.shape[2]//2,0,0))
    x = F.conv3d(x, ky, padding=(0, ky.shape[3]//2,0))
    x = F.conv3d(x, kx, padding=(0,0,kx.shape[4]//2))

    lap = (
        -6*x
        + F.pad(x[:,:,1:], (0,0,0,0,0,1))
        + F.pad(x[:,:,:-1], (0,0,0,0,1,0))
        + F.pad(x[:,:,:,1:], (0,0,0,1,0,0))
        + F.pad(x[:,:,:, :-1], (0,0,1,0,0,0))
        + F.pad(x[:,:,:,:,1:], (0,1,0,0,0,0))
        + F.pad(x[:,:,:,:,:-1], (1,0,0,0,0,0))
    )

    lap *= (sz**2 + sy**2 + sx**2)
    return lap.squeeze().cpu().numpy()


def local_minima_3d(log_t, min_distance):
    dz, dy, dx = min_distance
    x = log_t[None,None]
    min_filt = -F.max_pool3d(
        -x,
        kernel_size=(2*dz+1, 2*dy+1, 2*dx+1),
        stride=1,
        padding=(dz, dy, dx)
    )
    mask = (x == min_filt) & (x < 0)
    return mask.squeeze().nonzero(as_tuple=False)


# ============================================================
# Spot statistics
# ============================================================
def spot_statistics(img, coords, radius):
    """
    Compute 3D sum intensity and radii for each candidate spot.
    """
    Z, Y, X = img.shape
    intensities = []
    radii = []
    for z, y, x in coords:
        z1, z2 = max(0, z-radius), min(Z, z+radius+1)
        y1, y2 = max(0, y-radius), min(Y, y+radius+1)
        x1, x2 = max(0, x-radius), min(X, x+radius+1)
        sub = img[z1:z2, y1:y2, x1:x2]
        intensities.append(sub.sum())

        zz, yy, xx = np.indices(sub.shape)
        w = sub + 1e-6
        sz = np.sqrt(np.average((zz - zz.mean())**2, weights=w))
        sy = np.sqrt(np.average((yy - yy.mean())**2, weights=w))
        sx = np.sqrt(np.average((xx - xx.mean())**2, weights=w))
        radii.append([sz, sy, sx])

    return np.array(intensities), np.array(radii)


# ============================================================
# Gaussian fitting
# ============================================================
def gaussian_3d(coords, amp, z0, y0, x0, sz, sy, sx):
    z, y, x = coords
    return amp * np.exp(
        -( (z-z0)**2/(2*sz**2) + (y-y0)**2/(2*sy**2) + (x-x0)**2/(2*sx**2) )
    ).ravel()


def gaussian_fit_subset(img, coords, radius, expected_sigma, r2_threshold=0.8, seed=0):
    """
    Fit 3D Gaussian to each spot, return good vs bad spots based on R^2.
    """
    np.random.seed(seed)
    good_coords, bad_coords, good_intensities = [], [], []

    Z, Y, X = img.shape
    for z, y, x in coords:
        z1, z2 = max(0, z-radius), min(Z, z+radius+1)
        y1, y2 = max(0, y-radius), min(Y, y+radius+1)
        x1, x2 = max(0, x-radius), min(X, x+radius+1)
        sub = img[z1:z2, y1:y2, x1:x2]
        zz, yy, xx = np.indices(sub.shape)
        try:
            p0 = [sub.max(), *np.array(sub.shape)//2, *expected_sigma]
            popt, _ = curve_fit(gaussian_3d, (zz, yy, xx), sub.ravel(), p0=p0, maxfev=200)
            residuals = sub.ravel() - gaussian_3d((zz, yy, xx), *popt)
            r2 = 1 - np.sum(residuals**2)/np.sum((sub.ravel()-sub.mean())**2)
            if r2 >= r2_threshold:
                good_coords.append([z, y, x])
                good_intensities.append(sub.sum())
            else:
                bad_coords.append([z, y, x])
        except:
            bad_coords.append([z, y, x])

    return np.array(good_intensities), np.array(good_coords), np.array(bad_coords)


# ============================================================
# Raj lab plateau thresholding (3D sum version)
# ============================================================
def plateau_threshold_3D_sum(img, coords, radius=2, n_steps=50, pmin=90, pmax=99.99, save_plot=None):
    """
    Plateau/elbow thresholding based on 3D sum intensity around each spot.
    """
    Z, Y, X = img.shape
    spot_sums = []
    for z, y, x in coords:
        z1, z2 = max(0, z-radius), min(Z, z+radius+1)
        y1, y2 = max(0, y-radius), min(Y, y+radius+1)
        x1, x2 = max(0, x-radius), min(X, x+radius+1)
        sub = img[z1:z2, y1:y2, x1:x2]
        spot_sums.append(sub.sum())
    spot_sums = np.array(spot_sums)

    t_lo = np.percentile(spot_sums, pmin)
    t_hi = np.percentile(spot_sums, pmax)
    thresholds = np.linspace(t_lo, t_hi, n_steps)
    counts = np.array([np.sum(spot_sums <= t) for t in thresholds])

    d2 = np.gradient(np.gradient(counts))
    elbow_idx = np.argmax(d2)
    threshold = thresholds[elbow_idx]

    if save_plot:
        plt.figure()
        plt.plot(thresholds, counts, label="Spot count (3D sum)")
        plt.axvline(threshold, color='r', linestyle='--', label=f"Threshold={threshold:.2f}")
        plt.xlabel("3D sum intensity")
        plt.ylabel("Number of spots")
        plt.legend()
        plt.savefig(save_plot)
        plt.close()

    return threshold, spot_sums


# ============================================================
# Full GPU spot detector
# ============================================================
def detect_spots_gpu_full(image_np, sigma, min_distance, gaussian_radius=2,
                          gaussian_fit_fraction=1.0, r2_threshold=0.8,
                          random_seed=0, device="cuda"):
    """
    Complete pipeline:
    1) LoG minima
    2) Raj plateau threshold using 3D sum
    3) Gaussian fitting filter
    Returns: coords_final, threshold_used, log_img, sum_intensities, radii, good_coords, bad_coords
    """
    torch.manual_seed(random_seed)
    coords_final = np.empty((0,3), dtype=np.int16)
    good_coords = None
    bad_coords = None
    sum_intensities = np.array([])
    radii = np.array([])

    # --- LoG ---
    log_img = -log_filter_gpu(image_np, sigma, device)
    log_t = torch.from_numpy(log_img).to(device)

    # --- Local minima detection ---
    coords = local_minima_3d(log_t, min_distance).cpu().numpy().astype(np.int16)
    if len(coords) == 0:
        return coords_final, None, log_img, sum_intensities, radii, good_coords, bad_coords

    # --- Raj plateau threshold using 3D sum ---
    threshold_used, spot_sums = plateau_threshold_3D_sum(log_img, coords, radius=gaussian_radius)
    coords_thresh = coords[spot_sums <= threshold_used]

    # --- Compute spot statistics ---
    sum_intensities, radii = spot_statistics(image_np, coords_thresh, gaussian_radius)

    # --- Optional Gaussian fit ---
    if gaussian_fit_fraction > 0:
        _, good_coords, bad_coords = gaussian_fit_subset(image_np, coords_thresh, gaussian_radius, sigma, r2_threshold)
        coords_final = good_coords
    else:
        coords_final = coords_thresh

    return coords_final, threshold_used, log_img, sum_intensities, radii, good_coords, bad_coords


# ============================================================
# Visualization
# ============================================================
def plot_spot_example(img, coord, radius, save_path, title="Spot"):
    z, y, x = coord
    Z, Y, X = img.shape
    z1, z2 = max(0, z-radius), min(Z, z+radius+1)
    y1, y2 = max(0, y-radius), min(Y, y+radius+1)
    x1, x2 = max(0, x-radius), min(X, x+radius+1)
    sub = img[z1:z2, y1:y2, x1:x2]

    plt.figure(figsize=(9,3))
    plt.suptitle(title)
    plt.subplot(1,3,1)
    plt.imshow(sub[sub.shape[0]//2], cmap="hot"); plt.title("XY")
    plt.subplot(1,3,2)
    plt.imshow(sub[:, sub.shape[1]//2, :], cmap="hot"); plt.title("XZ")
    plt.subplot(1,3,3)
    plt.imshow(sub[:, :, sub.shape[2]//2], cmap="hot"); plt.title("YZ")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ============================================================
# GPU performance tuning
# ============================================================
def set_max_performance():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
