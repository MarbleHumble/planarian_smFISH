"""
GPU-native smFISH spot detection (Big-FISH replacement)
Author: Elias Guan

This module replaces:
- bigfish.stack.log_filter
- bigfish.detection.detect_spots

Designed for NVIDIA GPUs (A100 tested).
"""

import torch
import torch.nn.functional as F
import numpy as np


# -----------------------------
# Gaussian / LoG utilities
# -----------------------------

def gaussian_kernel_1d(sigma, device):
    radius = int(3 * sigma)
    x = torch.arange(-radius, radius + 1, device=device)
    kernel = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


def log_filter_gpu(image_np, sigma, device="cuda"):
    """
    3D Laplacian-of-Gaussian filter on GPU.
    Args:
        image_np (np.ndarray): ZYX
        sigma (tuple): (z, y, x) in pixels
    Returns:
        np.ndarray: LoG filtered image
    """
    device = torch.device(device)

    x = torch.from_numpy(image_np).float().to(device)
    x = x.unsqueeze(0).unsqueeze(0)  # (1,1,Z,Y,X)

    sz, sy, sx = sigma

    kz = gaussian_kernel_1d(sz, device).view(1, 1, -1, 1, 1)
    ky = gaussian_kernel_1d(sy, device).view(1, 1, 1, -1, 1)
    kx = gaussian_kernel_1d(sx, device).view(1, 1, 1, 1, -1)

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

    return lap.squeeze().cpu().numpy()


# -----------------------------
# Local maxima detection
# -----------------------------

def local_maxima_3d(log_img, min_distance):
    """
    GPU local maxima detection.
    Args:
        log_img (torch.Tensor): (Z,Y,X)
        min_distance (tuple): (dz, dy, dx)
    Returns:
        Tensor (N,3): spot coordinates
    """
    dz, dy, dx = min_distance

    x = log_img.unsqueeze(0).unsqueeze(0)
    max_filt = F.max_pool3d(
        x,
        kernel_size=(2 * dz + 1, 2 * dy + 1, 2 * dx + 1),
        stride=1,
        padding=(dz, dy, dx),
    )

    peaks = (x == max_filt) & (x > 0)
    coords = peaks.squeeze().nonzero(as_tuple=False)
    return coords


# -----------------------------
# Full GPU smFISH detector
# -----------------------------

def detect_spots_gpu(
    image_np,
    sigma,
    min_distance,
    threshold=None,
    auto_percentile=0.999,
    device="cuda",
):
    """
    Full GPU smFISH spot detection.

    Args:
        image_np (np.ndarray): ZYX
        sigma (tuple): LoG kernel size
        min_distance (tuple): minimum spot spacing
        threshold (float or None)
        auto_percentile (float): percentile for auto threshold
        device (str)

    Returns:
        spots (np.ndarray): (N,3)
        threshold_used (float)
        log_img (np.ndarray)
    """
    device = torch.device(device)

    log_img_np = log_filter_gpu(image_np, sigma, device)
    log_img = torch.from_numpy(log_img_np).to(device)

    if threshold is None:
        threshold = torch.quantile(log_img, auto_percentile).item()

    log_img = torch.where(log_img >= threshold, log_img, torch.zeros_like(log_img))

    coords = local_maxima_3d(log_img, min_distance)

    return coords.cpu().numpy(), threshold, log_img_np


# -----------------------------
# Utility
# -----------------------------

def set_max_performance():
    """Enable maximum CUDA performance."""
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
