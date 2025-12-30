"""
GPU-native smFISH spot detection pipeline (Tissue-optimized)

Pipeline:
1) 3D LoG minima detection (GPU)
2) Depth percentile filtering
3) Local contrast filtering
4) Raj plateau integrated intensity
5) Moment-based size & aspect ratio filtering
6) (Optional) Gaussian fitting for localization only

Author: Elias Guan
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# ============================================================
# ------------------- LoG Utilities --------------------------
# ============================================================

def gaussian_kernel_1d(sigma, device):
    radius = max(1, int(3 * sigma))
    x = torch.arange(-radius, radius + 1, device=device)
    k = torch.exp(-(x**2) / (2 * sigma**2))
    return k / k.sum()

def log_filter_gpu(image_np, sigma, device="cuda"):
    """
    3D LoG filtering using separable Gaussian + discrete Laplacian
    """
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
        + torch.roll(x, 1, 2) + torch.roll(x, -1, 2)
        + torch.roll(x, 1, 3) + torch.roll(x, -1, 3)
        + torch.roll(x, 1, 4) + torch.roll(x, -1, 4)
    )

    lap *= (sz**2 + sy**2 + sx**2)
    return (-lap).squeeze().cpu().numpy()

def local_minima_3d_strict(log_img, min_distance, depth_percentile=0.01, device="cuda"):
    """
    Strict 3D local minima detection (GPU)
    Only keep minima that exceed depth percentile.
    """
    dz, dy, dx = min_distance
    x = torch.from_numpy(log_img).to(device)[None, None]
    min_filt = -F.max_pool3d(
        -x, kernel_size=(2*dz+1, 2*dy+1, 2*dx+1),
        stride=1, padding=(dz, dy, dx)
    )
    depth_thresh = np.percentile(log_img, depth_percentile)
    mask = (x == min_filt) & (x < -depth_thresh)
    coords = mask.squeeze().nonzero(as_tuple=False).cpu().numpy().astype(np.int16)
    return coords

# ============================================================
# ---------------- Spot metrics ------------------------------
# ============================================================

def compute_log_depths(log_img, coords, radius=2):
    depths = np.zeros(len(coords), np.float32)
    Z, Y, X = log_img.shape
    for i, (z, y, x) in enumerate(coords):
        z1, z2 = max(0, z-radius), min(Z, z+radius+1)
        y1, y2 = max(0, y-radius), min(Y, y+radius+1)
        x1, x2 = max(0, x-radius), min(X, x+radius+1)
        patch = log_img[z1:z2, y1:y2, x1:x2]
        bg = np.percentile(patch, 90)
        depths[i] = bg - log_img[z, y, x]
    return depths

def compute_local_contrast(img, coords, radius):
    Z, Y, X = img.shape
    out = np.zeros(len(coords), dtype=np.float32)
    for i, (z, y, x) in enumerate(coords):
        z1, z2 = max(0, z-radius), min(Z, z+radius+1)
        y1, y2 = max(0, y-radius), min(Y, y+radius+1)
        x1, x2 = max(0, x-radius), min(X, x+radius+1)
        patch = img[z1:z2, y1:y2, x1:x2]
        center = img[z, y, x]
        bg = np.percentile(patch, 75)
        out[i] = (center + 1e-6) / (bg + 1e-6)
    return out

def spot_statistics(img, coords, radius=2):
    Z, Y, X = img.shape
    intensities = np.zeros(len(coords), np.float32)
    moments = np.zeros((len(coords), 3), np.float32)
    for i, (z, y, x) in enumerate(coords):
        z1, z2 = max(0, z-radius), min(Z, z+radius+1)
        y1, y2 = max(0, y-radius), min(Y, y+radius+1)
        x1, x2 = max(0, x-radius), min(X, x+radius+1)
        sub = img[z1:z2, y1:y2, x1:x2]
        intensities[i] = sub.sum()
        zz, yy, xx = np.indices(sub.shape)
        w = sub + 1e-6
        moments[i, 0] = np.sqrt(np.average((zz-zz.mean())**2, weights=w))
        moments[i, 1] = np.sqrt(np.average((yy-yy.mean())**2, weights=w))
        moments[i, 2] = np.sqrt(np.average((xx-xx.mean())**2, weights=w))
    return intensities, moments

def raj_plateau_threshold(intensities, smooth_window=11, slope_thresh=0.02):
    I = np.sort(intensities)[::-1]
    I_s = uniform_filter1d(I, smooth_window)
    dI = np.abs(np.gradient(I_s))
    dI /= dI.max()
    idx = np.where(dI < slope_thresh)[0]
    cut = idx[0] if len(idx) else int(0.2 * len(I))
    return I[cut], cut

def _save_hist(values, xlabel, folder, fname):
    plt.figure(figsize=(4,3))
    plt.hist(values, bins=50)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, fname))
    plt.close()

# ============================================================
# ---------------- Main pipeline -----------------------------
# ============================================================

def detect_spots_gpu(
    image_np,
    sigma,
    min_distance,
    radius=2,
    depth_percentile=99.5,
    min_contrast=2.0,
    size_bounds=(0.6, 3.0),
    aspect_ratio_max=2.5,
    device="cuda",
    diagnostics=None,
    # --- NEW CONFIGURABLE INTENSITY FILTERING OPTIONS ---
    use_raj_plateau=True,
    raj_slope_thresh=0.02,
    raj_smooth_window=11,
    intensity_percentile=None,
):
    """
    GPU smFISH detection with diagnostics.
    Returns:
        coords_final (N,3)
        stats (dict)
    """

    stats = {}

    # 1. LoG
    log_img = -log_filter_gpu(image_np, sigma, device=device)

    # 2. Local minima
    coords = local_minima_3d_strict(
        log_img,
        min_distance=min_distance,
        depth_percentile=depth_percentile,
        device=device
    )
    stats["n_minima"] = len(coords)
    if len(coords) == 0:
        return np.empty((0, 3)), stats

    # 3. Depth filter
    depths = compute_log_depths(log_img, coords, radius=radius)
    depth_thresh = np.percentile(depths, 50)
    keep = depths >= depth_thresh
    coords = coords[keep]
    depths = depths[keep]
    stats["n_after_depth"] = len(coords)

    # 4. Local contrast filter
    contrasts = compute_local_contrast(image_np, coords, radius)
    keep = contrasts >= min_contrast
    coords = coords[keep]
    contrasts = contrasts[keep]
    stats["n_after_contrast"] = len(coords)

    # 5. Raj plateau / intensity-based filtering
    intensities, moments = spot_statistics(image_np, coords, radius)

    if use_raj_plateau:
        # Classic Raj plateau heuristic
        I_thresh, _ = raj_plateau_threshold(
            intensities,
            smooth_window=raj_smooth_window,
            slope_thresh=raj_slope_thresh,
        )
    elif intensity_percentile is not None:
        # Simple percentile-based threshold on integrated intensity
        I_thresh = np.percentile(intensities, intensity_percentile)
    else:
        # Skip intensity filtering entirely
        I_thresh = intensities.min() - 1  # everything passes

    keep = intensities >= I_thresh
    coords = coords[keep]
    moments = moments[keep]
    stats["n_after_raj"] = len(coords)

    # 6. Moment-based size & anisotropy filter
    sz, sy, sx = moments.T
    size_um = np.cbrt(sz * sy * sx)
    aspect_ratio = np.maximum.reduce([sy / (sx + 1e-6), sx / (sy + 1e-6)])
    keep = (
        (size_um >= size_bounds[0]) &
        (size_um <= size_bounds[1]) &
        (aspect_ratio <= aspect_ratio_max)
    )
    coords_final = coords[keep]
    stats["n_after_size"] = len(coords_final)

    # Diagnostics
    if diagnostics is not None:
        os.makedirs(diagnostics, exist_ok=True)
        _save_hist(depths, "LoG depth", diagnostics, "depth.png")
        _save_hist(contrasts, "Local contrast", diagnostics, "contrast.png")
        _save_hist(intensities, "Integrated intensity", diagnostics, "intensity.png")
        _save_hist(size_um, "Moment-derived size (µm)", diagnostics, "size.png")

    return coords_final, stats


# ============================================================
# ---------------- Big-FISH Protocol Replication -------------
# ============================================================

def detect_spots_gpu_bigfish(
    image_np,
    sigma,
    min_distance,
    threshold=None,
    voxel_size=None,
    spot_radius=None,
    device="cuda",
    return_threshold=True,
):
    """
    GPU implementation that exactly replicates Big-FISH detect_spots algorithm.
    
    Big-FISH algorithm (simplified):
    1. Apply LoG filter to image
    2. Find local minima in LoG image (with minimum_distance constraint)
    3. Apply threshold filtering on LoG values
    4. Return filtered spots
    
    This replicates Big-FISH's simple approach without extra filtering stages.
    
    Args:
        image_np: Input 3D image array (Z, Y, X)
        sigma: LoG kernel size (z, y, x) - same as log_kernel_size in Big-FISH
        min_distance: Minimum distance between spots (z, y, x) - same as minimum_distance
        threshold: LoG threshold value. If None, will use a percentile-based auto-threshold.
                   In Big-FISH, lower (more negative) LoG values = stronger spots.
        voxel_size: Not used in detection but kept for API compatibility with Big-FISH
        spot_radius: Not used in detection but kept for API compatibility with Big-FISH
        device: GPU device ("cuda" or "cpu")
        return_threshold: Whether to return the threshold used
        
    Returns:
        spots: Array of spot coordinates (N, 3) as int16
        threshold_used: Threshold value (if return_threshold=True)
    """
    device = torch.device(device)
    
    # ============================================================
    # STEP 1: Apply LoG filter (exactly like Big-FISH)
    # ============================================================
    # log_filter_gpu returns negative LoG (spots are negative in standard LoG)
    log_img = log_filter_gpu(image_np, sigma, device=device)
    
    # Big-FISH works with LoG directly (spots are negative values = local minima)
    # We need to negate to work with maxima detection (then negate threshold logic)
    log_img_neg = -log_img  # Now spots are positive = easier to work with
    
    # ============================================================
    # STEP 2: Find local minima in LoG image (exactly like Big-FISH)
    # ============================================================
    dz, dy, dx = min_distance
    log_tensor = torch.from_numpy(log_img_neg).float().to(device)[None, None]
    
    # Local minima detection: find points that are maxima in negated space
    # (which are minima in original LoG space)
    max_filt = F.max_pool3d(
        log_tensor,
        kernel_size=(2*dz+1, 2*dy+1, 2*dx+1),
        stride=1,
        padding=(dz, dy, dx)
    )
    
    # Points that are local maxima in negated space = minima in original LoG
    is_maxima = (log_tensor == max_filt)
    
    # Get coordinates of all local minima (in original LoG space)
    coords_tensor = is_maxima.squeeze().nonzero(as_tuple=False)
    coords = coords_tensor.cpu().numpy().astype(np.int16)
    
    if len(coords) == 0:
        if return_threshold:
            return np.empty((0, 3), dtype=np.int16), None
        return np.empty((0, 3), dtype=np.int16)
    
    # ============================================================
    # STEP 3: Apply threshold filtering (exactly like Big-FISH)
    # ============================================================
    # Get LoG values at minima coordinates (use original LoG, negative values)
    log_values_original = log_img[coords[:, 0], coords[:, 1], coords[:, 2]]
    
    if threshold is None:
        # Auto-threshold: Big-FISH keeps spots with more negative LoG values (stronger signals)
        # Use percentile on original LoG values (negative = stronger)
        # Lower percentile = more negative values = stronger spots
        threshold = np.percentile(log_values_original, 5.0)  # Keep bottom 5% (most negative = strongest)
    else:
        # Threshold is provided (should be negative for Big-FISH convention)
        # If positive threshold provided, assume it needs negation
        if threshold > 0:
            threshold = -threshold
    
    # Keep spots where original LoG value is below threshold (more negative = stronger)
    keep = log_values_original <= threshold
    spots = coords[keep]
    
    if return_threshold:
        return spots, threshold
    
    return spots
