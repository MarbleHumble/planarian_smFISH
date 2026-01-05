"""
Improved spot detection with multi-stage filtering for high-background images.

This module provides an enhanced detection algorithm that adds filtering stages
beyond simple threshold-based detection to better handle tissue images with
high background and variable spot brightness.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import uniform_filter1d

try:
    from .gpu_smfish_v2 import (
        log_filter_gpu, 
        compute_local_contrast, 
        spot_statistics,
        local_minima_3d_strict
    )
    HAS_GPU_FUNCS = True
except ImportError:
    HAS_GPU_FUNCS = False
    print("Warning: GPU functions not available, will use CPU fallback")


def raj_plateau_threshold(intensities, smooth_window=21, slope_thresh=0.05):
    """
    Raj plateau thresholding - finds the "plateau" in sorted intensities
    where the slope changes, indicating the transition from real spots to noise.
    """
    I = np.sort(intensities)[::-1]  # Sort descending
    I_s = uniform_filter1d(I.astype(float), smooth_window)
    dI = np.abs(np.gradient(I_s))
    
    # Find first point where slope is below threshold
    for i in range(len(dI)):
        if dI[i] < slope_thresh * I_s[0]:
            return I[i] if i > 0 else I[0]
    
    return I[-1]  # Fallback: use minimum


def detect_spots_improved(
    image_np,
    sigma=(1, 1.7, 1.7),
    min_distance=(3, 3, 3),
    threshold=None,  # LoG threshold (optional initial filter)
    min_contrast=None,  # Local contrast threshold (None = skip this filter)
    use_intensity_filter=True,  # Use Raj plateau for intensity filtering
    intensity_percentile=None,  # Alternative: use percentile instead of Raj plateau
    min_size_um=0.3,  # Minimum spot size (µm) - filters tiny noise
    max_size_um=4.0,  # Maximum spot size (µm) - filters large artifacts
    aspect_ratio_max=3.0,  # Maximum aspect ratio (filters elongated objects)
    voxel_size_um=(0.361, 0.075, 0.075),  # Voxel size in µm (z, y, x)
    device="cuda",
    return_statistics=False,
):
    """
    Improved spot detection with multi-stage filtering.
    
    Algorithm:
    1. LoG filter
    2. Local minima detection
    3. (Optional) Initial LoG threshold filter
    4. Local contrast filter - removes spots that don't stand out from local background
    5. Intensity filter (Raj plateau or percentile) - removes weak spots
    6. Size & shape filter - removes oddly sized/shaped objects
    
    Args:
        image_np: Input 3D image (Z, Y, X)
        sigma: LoG kernel size
        min_distance: Minimum distance between spots
        threshold: Optional initial LoG threshold (if None, skip this stage)
        min_contrast: Minimum local contrast ratio (spot/local background)
        use_intensity_filter: Use Raj plateau for intensity filtering
        intensity_percentile: Alternative: percentile threshold (0-100)
        min_size_um: Minimum spot size in µm (filters tiny noise)
        max_size_um: Maximum spot size in µm (filters large artifacts)
        aspect_ratio_max: Maximum aspect ratio (filters elongated objects)
        device: "cuda" or "cpu"
        return_statistics: If True, return filtering statistics
        
    Returns:
        spots: (N, 3) array of spot coordinates
        stats: (optional) Dictionary with filtering statistics
    """
    if not HAS_GPU_FUNCS:
        raise ImportError("GPU functions required for improved detection")
    
    device = torch.device(device)
    stats = {}
    
    # STEP 1: LoG filter
    # log_filter_gpu returns negative LoG (spots are negative in standard LoG)
    log_img = log_filter_gpu(image_np, sigma, device=device)
    
    # STEP 2: Local minima detection
    # local_minima_3d_strict expects the LoG image (it works with negative values)
    # Based on detect_spots_gpu, we negate to work with maxima
    log_img_neg = -log_img  # Negate to work with maxima detection
    coords = local_minima_3d_strict(
        log_img_neg,
        min_distance=min_distance,
        depth_percentile=50.0,  # Keep top 50% of minima (much more lenient)
        device=device
    )
    stats["n_minima"] = len(coords)
    
    if len(coords) == 0:
        if return_statistics:
            return np.empty((0, 3), dtype=np.int16), stats
        return np.empty((0, 3), dtype=np.int16)
    
    # STEP 3: Optional initial LoG threshold filter
    if threshold is not None:
        # Use original (negative) LoG values for thresholding
        log_values_original = log_img[coords[:, 0], coords[:, 1], coords[:, 2]]
        # Threshold should be negative (lower = more negative = stronger spots)
        if threshold > 0:
            threshold = -threshold  # Convert to negative if needed
        keep = log_values_original <= threshold  # Keep spots with more negative LoG (stronger)
        coords = coords[keep]
        stats["n_after_log_thresh"] = len(coords)
    
    if len(coords) == 0:
        if return_statistics:
            return np.empty((0, 3), dtype=np.int16), stats
        return np.empty((0, 3), dtype=np.int16)
    
    # STEP 4: Local contrast filter (optional)
    # This helps for high-background images, but can be too aggressive
    radius = 2
    if min_contrast is not None:
        contrasts = compute_local_contrast(image_np, coords, radius)
        keep = contrasts >= min_contrast
        coords = coords[keep]
        contrasts = contrasts[keep]
        stats["n_after_contrast"] = len(coords)
        stats["mean_contrast"] = float(contrasts.mean()) if len(contrasts) > 0 else 0.0
    else:
        # Skip contrast filter - rely on intensity and size filters instead
        stats["n_after_contrast"] = len(coords)
        stats["mean_contrast"] = None
    
    if len(coords) == 0:
        if return_statistics:
            return np.empty((0, 3), dtype=np.int16), stats
        return np.empty((0, 3), dtype=np.int16)
    
    # STEP 5: Intensity filter (Raj plateau or percentile)
    intensities, moments = spot_statistics(image_np, coords, radius)
    
    if use_intensity_filter:
        if intensity_percentile is not None:
            I_thresh = np.percentile(intensities, intensity_percentile)
        else:
            # Use Raj plateau - adapts to the distribution of intensities
            I_thresh = raj_plateau_threshold(intensities, smooth_window=21, slope_thresh=0.05)
        
        keep = intensities >= I_thresh
        coords = coords[keep]
        intensities = intensities[keep]
        moments = moments[keep]
        stats["n_after_intensity"] = len(coords)
        stats["intensity_threshold"] = float(I_thresh)
    else:
        stats["n_after_intensity"] = len(coords)
        stats["intensity_threshold"] = None
    
    if len(coords) == 0:
        if return_statistics:
            return np.empty((0, 3), dtype=np.int16), stats
        return np.empty((0, 3), dtype=np.int16)
    
    # STEP 6: Size & shape filter
    # Compute spot sizes from second moments
    sz, sy, sx = moments.T
    
    # Convert to µm using provided voxel_size_um
    size_z_um = sz * voxel_size_um[0]
    size_y_um = sy * voxel_size_um[1]
    size_x_um = sx * voxel_size_um[2]
    
    # Geometric mean size
    size_um = np.cbrt(size_z_um * size_y_um * size_x_um)
    
    # Aspect ratio (max ratio of any two dimensions)
    aspect_ratios = np.maximum.reduce([
        size_z_um / (size_y_um + 1e-6),
        size_z_um / (size_x_um + 1e-6),
        size_y_um / (size_x_um + 1e-6),
        size_y_um / (size_z_um + 1e-6),
        size_x_um / (size_y_um + 1e-6),
        size_x_um / (size_z_um + 1e-6),
    ])
    
    keep = (
        (size_um >= min_size_um) &
        (size_um <= max_size_um) &
        (aspect_ratios <= aspect_ratio_max)
    )
    coords_final = coords[keep]
    stats["n_after_size"] = len(coords_final)
    stats["mean_size_um"] = float(size_um[keep].mean()) if np.any(keep) else 0.0
    
    if return_statistics:
        return coords_final.astype(np.int16), stats
    
    return coords_final.astype(np.int16)

