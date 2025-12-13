"""
GPU-native smFISH spot detection (Big-FISH equivalent)
Author: Elias Guan

Returns:
    coords (np.ndarray): spot coordinates (N,3)
    threshold (float)
    log_img (np.ndarray)
    sum_intensities (np.ndarray)
    radii (np.ndarray): σz, σy, σx per spot
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os

# -----------------------------
# Gaussian / LoG utilities
# -----------------------------
def gaussian_kernel_1d(sigma, device):
    radius = max(1, int(3*sigma))
    x = torch.arange(-radius, radius+1, device=device)
    kernel = torch.exp(-(x**2)/(2*sigma**2))
    kernel /= kernel.sum()
    return kernel

def log_filter_gpu(image_np, sigma, device="cuda"):
    """3D Laplacian-of-Gaussian filter on GPU."""
    device = torch.device(device)
    x = torch.from_numpy(image_np).float().to(device).unsqueeze(0).unsqueeze(0)

    sz, sy, sx = sigma
    kz = gaussian_kernel_1d(sz, device).view(1,1,-1,1,1)
    ky = gaussian_kernel_1d(sy, device).view(1,1,1,-1,1)
    kx = gaussian_kernel_1d(sx, device).view(1,1,1,1,-1)

    x = F.conv3d(x, kz, padding=(kz.shape[2]//2,0,0))
    x = F.conv3d(x, ky, padding=(0,ky.shape[3]//2,0))
    x = F.conv3d(x, kx, padding=(0,0,kx.shape[4]//2))

    lap_kernel = torch.tensor([[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]],
                              dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    lap = F.conv3d(x, lap_kernel, padding=1)
    lap *= sz**2 + sy**2 + sx**2
    return lap.squeeze().cpu().numpy()

# -----------------------------
# Local minima detection
# -----------------------------
def local_minima_3d(log_img, min_distance):
    dz, dy, dx = min_distance
    x = log_img.unsqueeze(0).unsqueeze(0)
    min_filt = -F.max_pool3d(-x, kernel_size=(2*dz+1,2*dy+1,2*dx+1), stride=1, padding=(dz,dy,dx))
    peaks = (x == min_filt) & (x < 0)
    return peaks.squeeze().nonzero(as_tuple=False)

# -----------------------------
# Gaussian patch validation
# -----------------------------
def filter_gaussian_spots(img, coords, radius, expected_sigma, r2_threshold=0.8):
    """Keep only spots that fit a 3D Gaussian, compute sum intensities and radii."""
    def gaussian_3d(coords, amp, z0, y0, x0, sz, sy, sx):
        z, y, x = coords
        return amp * np.exp(-((z-z0)**2/(2*sz**2)+(y-y0)**2/(2*sy**2)+(x-x0)**2/(2*sx**2))).ravel())

    kept_coords, sum_intensities, radii = [], [], []
    Z,Y,X = img.shape
    for z,y,x in coords:
        z1,z2 = max(0,z-radius), min(Z,z+radius+1)
        y1,y2 = max(0,y-radius), min(Y,y+radius+1)
        x1,x2 = max(0,x-radius), min(X,x+radius+1)
        sub = img[z1:z2, y1:y2, x1:x2]
        zz,yy,xx = np.meshgrid(np.arange(sub.shape[0]),
                               np.arange(sub.shape[1]),
                               np.arange(sub.shape[2]), indexing='ij')
        try:
            p0 = [sub.max(), sub.shape[0]//2, sub.shape[1]//2, sub.shape[2]//2, *expected_sigma]
            popt,_ = curve_fit(gaussian_3d, (zz,yy,xx), sub.ravel(), p0=p0)
            residuals = sub.ravel()-gaussian_3d((zz,yy,xx),*popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((sub.ravel()-sub.mean())**2)
            r2 = 1 - ss_res/ss_tot
            if r2 >= r2_threshold:
                kept_coords.append([z,y,x])
                sum_intensities.append(sub.sum())
                radii.append([popt[4], popt[5], popt[6]])
        except Exception:
            continue
    return np.array(kept_coords), np.array(sum_intensities), np.array(radii)

# -----------------------------
# Full GPU detector
# -----------------------------
def detect_spots_gpu(
    image_np,
    sigma,
    min_distance,
    threshold=None,
    auto_percentile=0.999,
    gaussian_radius=2,
    r2_threshold=0.8,
    spots_radius_detection=True,
    device="cuda"
):
    device = torch.device(device)
    log_img_np = -log_filter_gpu(image_np, sigma, device)
    log_img = torch.from_numpy(log_img_np).to(device)

    if threshold is None:
        threshold = torch.quantile(log_img[log_img<0], auto_percentile).item()
    log_img = torch.where(log_img <= threshold, log_img, torch.zeros_like(log_img))
    coords = local_minima_3d(log_img, min_distance)

    if spots_radius_detection:
        coords, sum_intensities, radii = filter_gaussian_spots(image_np, coords, gaussian_radius, sigma, r2_threshold)
    else:
        sum_intensities = np.zeros(len(coords))
        radii = np.zeros((len(coords),3))

    return coords, threshold, log_img_np, sum_intensities, radii

# -----------------------------
# Spot visualization helper
# -----------------------------
def plot_spot_example(img, coord, gaussian_fit=True, radius=2, expected_sigma=[1,1,1], save_path="spot_example.png"):
    """
    Plot 2D slices (XY, XZ, YZ) of a spot with optional Gaussian fit.
    """
    z,y,x = coord
    Z,Y,X = img.shape
    z1,z2 = max(0,z-radius), min(Z,z+radius+1)
    y1,y2 = max(0,y-radius), min(Y,y+radius+1)
    x1,x2 = max(0,x-radius), min(X,x+radius+1)
    sub = img[z1:z2, y1:y2, x1:x2]

    plt.figure(figsize=(9,3))
    plt.suptitle(f"Spot at {coord} - {'Gaussian' if gaussian_fit else 'Discarded'}")

    plt.subplot(1,3,1)
    plt.imshow(sub[sub.shape[0]//2], cmap='hot')
    plt.title("XY slice")

    plt.subplot(1,3,2)
    plt.imshow(sub[:,sub.shape[1]//2,:], cmap='hot')
    plt.title("XZ slice")

    plt.subplot(1,3,3)
    plt.imshow(sub[:,:,sub.shape[2]//2], cmap='hot')
    plt.title("YZ slice")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# -----------------------------
# Performance
# -----------------------------
def set_max_performance():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")