#!/usr/bin/env python3
"""
Test threshold validation using IMPROVED spot detection with multi-stage filtering.

This version uses the improved detection algorithm that filters background noise
while preserving real spots, making it better for high-background tissue images.
"""

import numpy as np
from tifffile import imread, imwrite
from pathlib import Path
import json
import time
import argparse
import yaml
from scipy.ndimage import gaussian_filter

try:
    from functions.improved_spot_detection import detect_spots_improved
    HAS_IMPROVED = True
except ImportError:
    HAS_IMPROVED = False
    print("Error: Improved detection function not available")
    print("Make sure functions/improved_spot_detection.py exists")
    exit(1)


def find_spot_dense_region(image, region_size=1024, n_candidates=10):
    """
    Find a region with high spot density by doing quick detection on the full image.
    
    Returns:
        cropped_image, region_coords dict
    """
    z, y, x = image.shape
    
    # Quick spot detection on full image with very permissive settings
    from functions.gpu_smfish_v2 import log_filter_gpu, local_minima_3d_strict
    import torch
    
    print(f"  Scanning full image for spot-dense regions...")
    device = torch.device("cpu")  # Use CPU for this quick scan
    
    # Quick LoG filter
    sigma = (1, 1.7, 1.7)
    log_img = log_filter_gpu(image, sigma, device=device)
    log_img_neg = -log_img
    
    # Find all local minima (very permissive)
    min_distance = (3, 3, 3)
    all_coords = local_minima_3d_strict(
        log_img_neg,
        min_distance=min_distance,
        depth_percentile=50.0,  # Keep top 50%
        device=device
    )
    
    print(f"  Found {len(all_coords)} candidate spots in full image")
    
    if len(all_coords) == 0:
        # Fallback to random if no spots found
        print("  No spots found, using random region")
        return extract_random_region(image, region_size)
    
    # Project spots to 2D (sum over Z) to find dense regions
    spot_density = np.zeros((y, x), dtype=np.float32)
    for z_coord, y_coord, x_coord in all_coords:
        if 0 <= y_coord < y and 0 <= x_coord < x:
            spot_density[y_coord, x_coord] += 1
    
    # Smooth the density map
    spot_density_smooth = gaussian_filter(spot_density, sigma=region_size//4)
    
    # Find top N candidate regions
    step = region_size // 2  # Overlap regions for better coverage
    candidates = []
    
    for y_start in range(0, y - region_size + 1, step):
        for x_start in range(0, x - region_size + 1, step):
            y_end = min(y_start + region_size, y)
            x_end = min(x_start + region_size, x)
            
            # Calculate average spot density in this region
            density_sum = spot_density_smooth[y_start:y_end, x_start:x_end].sum()
            candidates.append((density_sum, y_start, y_end, x_start, x_end))
    
    # Sort by density (highest first)
    candidates.sort(reverse=True)
    
    # Pick the best region
    _, y_start, y_end, x_start, x_end = candidates[0]
    
    print(f"  Selected region with density score: {candidates[0][0]:.1f}")
    print(f"  Region: Y={y_start}-{y_end}, X={x_start}-{x_end}")
    
    cropped = image[:, y_start:y_end, x_start:x_end]
    
    coords = {
        'z_start': 0,
        'z_end': z,
        'y_start': y_start,
        'y_end': y_end,
        'x_start': x_start,
        'x_end': x_end,
        'region_size': region_size
    }
    
    return cropped, coords


def extract_random_region(image, region_size=1024):
    """
    Extract a random 1024x1024 region (keeping full Z).
    Fallback function if spot-dense region finding fails.
    """
    z, y, x = image.shape
    
    # Random start positions (with margin)
    y_start = np.random.randint(0, max(1, y - region_size))
    x_start = np.random.randint(0, max(1, x - region_size))
    y_end = min(y_start + region_size, y)
    x_end = min(x_start + region_size, x)
    
    # Adjust if region is smaller than requested
    if y_end - y_start < region_size:
        y_start = max(0, y - region_size)
        y_end = y
    if x_end - x_start < region_size:
        x_start = max(0, x - region_size)
        x_end = x
    
    cropped = image[:, y_start:y_end, x_start:x_end]
    
    coords = {
        'z_start': 0,
        'z_end': z,
        'y_start': y_start,
        'y_end': y_end,
        'x_start': x_start,
        'x_end': x_end,
        'region_size': region_size
    }
    
    return cropped, coords


def create_imagej_overlay(image_2d, spots_2d, output_path, spot_color=(1, 0, 0)):
    """
    Create an RGB overlay image for ImageJ (8-bit).
    
    Args:
        image_2d: 2D numpy array (single Z-slice)
        spots_2d: Nx2 array of (y, x) spot coordinates
        output_path: Path to save RGB TIFF
        spot_color: RGB tuple (0-1 range) for spot color, default red
    """
    # Normalize image to 0-1 range
    img_min = image_2d.min()
    img_max = image_2d.max()
    if img_max > img_min:
        img_norm = (image_2d - img_min) / (img_max - img_min)
    else:
        img_norm = image_2d.astype(np.float32)
    
    # Create RGB image (grayscale background)
    rgb = np.stack([img_norm, img_norm, img_norm], axis=-1)
    
    # Overlay spots
    if len(spots_2d) > 0:
        # Convert to integer coordinates
        y_coords = spots_2d[:, 0].astype(int)
        x_coords = spots_2d[:, 1].astype(int)
        
        # Filter valid coordinates
        valid = (
            (y_coords >= 0) & (y_coords < rgb.shape[0]) &
            (x_coords >= 0) & (x_coords < rgb.shape[1])
        )
        y_coords = y_coords[valid]
        x_coords = x_coords[valid]
        
        # Draw spots (3x3 pixel squares for visibility)
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                y_idx = y_coords + dy
                x_idx = x_coords + dx
                
                y_valid = (y_idx >= 0) & (y_idx < rgb.shape[0])
                x_valid = (x_idx >= 0) & (x_idx < rgb.shape[1])
                valid_mask = y_valid & x_valid
                
                if np.any(valid_mask):
                    rgb[y_idx[valid_mask], x_idx[valid_mask], :] = spot_color
    
    # Convert to uint8 for ImageJ (scale to 0-255)
    rgb_uint8 = (rgb * 255).astype(np.uint8)
    
    # Save as RGB TIFF (8-bit)
    imwrite(output_path, rgb_uint8, photometric='rgb')


def create_3d_overlay(image_3d, spots_3d, output_path, spot_color=(1, 0, 0)):
    """
    Create a 3D RGB overlay stack for ImageJ (8-bit).
    
    Args:
        image_3d: 3D numpy array [Z, Y, X]
        spots_3d: Nx3 array of (z, y, x) spot coordinates
        output_path: Path to save 3D RGB TIFF stack
        spot_color: RGB tuple (0-1 range) for spot color, default red
    """
    z, y, x = image_3d.shape
    
    # Normalize each Z-slice independently
    rgb_stack = np.zeros((z, y, x, 3), dtype=np.float32)
    
    for z_idx in range(z):
        img_slice = image_3d[z_idx]
        img_min = img_slice.min()
        img_max = img_slice.max()
        if img_max > img_min:
            img_norm = (img_slice - img_min) / (img_max - img_min)
        else:
            img_norm = img_slice.astype(np.float32)
        
        # Create RGB (grayscale background)
        rgb_stack[z_idx] = np.stack([img_norm, img_norm, img_norm], axis=-1)
    
    # Overlay spots in 3D
    if len(spots_3d) > 0:
        # Convert to integer coordinates
        z_coords = spots_3d[:, 0].astype(int)
        y_coords = spots_3d[:, 1].astype(int)
        x_coords = spots_3d[:, 2].astype(int)
        
        # Filter valid coordinates
        valid = (
            (z_coords >= 0) & (z_coords < z) &
            (y_coords >= 0) & (y_coords < y) &
            (x_coords >= 0) & (x_coords < x)
        )
        z_coords = z_coords[valid]
        y_coords = y_coords[valid]
        x_coords = x_coords[valid]
        
        # Draw spots (3x3x3 voxel cubes for visibility)
        for dz in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    z_idx = z_coords + dz
                    y_idx = y_coords + dy
                    x_idx = x_coords + dx
                    
                    z_valid = (z_idx >= 0) & (z_idx < z)
                    y_valid = (y_idx >= 0) & (y_idx < y)
                    x_valid = (x_idx >= 0) & (x_idx < x)
                    valid_mask = z_valid & y_valid & x_valid
                    
                    if np.any(valid_mask):
                        rgb_stack[z_idx[valid_mask], y_idx[valid_mask], x_idx[valid_mask], :] = spot_color
    
    # Convert to uint8 (scale to 0-255)
    rgb_uint8 = (rgb_stack * 255).astype(np.uint8)
    
    # Reshape to [Z, Y, X, 3] for tifffile (or [Z, 3, Y, X] for separate channels)
    # tifffile expects [T, Z, Y, X, C] or [Z, Y, X, C]
    imwrite(output_path, rgb_uint8, photometric='rgb')


def test_thresholds_improved(image_path, output_dir, region_size=1024,
                            thresholds=None, config_path=None,
                            sigma=None, min_distance=None,
                            min_contrast=1.5,
                            use_gpu=True, device="cuda",
                            slice_for_overlay='middle'):
    """
    Test thresholds using IMPROVED detection algorithm.
    
    Args:
        thresholds: List of LoG threshold values to test (None = auto-generate)
                   Note: With improved detection, threshold is optional initial filter
    """
    # Load parameters from config.yaml if provided
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        sigma = sigma or tuple(config.get('kernel_size', [1, 1.7, 1.7]))
        min_distance = min_distance or tuple(config.get('minimal_distance', [3, 3, 3]))
        print(f"  Loaded parameters from config.yaml")
        print(f"    sigma (kernel_size): {sigma}")
        print(f"    min_distance: {min_distance}")
        print(f"    min_contrast: {min_contrast} (improved detection parameter)")
    else:
        # Default values
        sigma = sigma or (1, 1.7, 1.7)
        min_distance = min_distance or (3, 3, 3)
    
    if thresholds is None:
        # Generate a range of thresholds to test
        # For improved detection, threshold is less critical, but we'll test a range
        thresholds = list(range(17))  # Will be converted to actual LoG thresholds
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create separate folder for 3D overlays
    output_3d_dir = output_dir / "3D_overlays"
    output_3d_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Processing with IMPROVED detection: {Path(image_path).name}")
    print(f"{'='*60}")
    
    # Load image
    print("Loading image...")
    start_load = time.time()
    image = imread(image_path)
    load_time = time.time() - start_load
    print(f"  Image shape: {image.shape}")
    print(f"  Load time: {load_time:.2f}s")
    
    # Extract region (keeps full Z-depth for 3D processing)
    print(f"\nExtracting {region_size}x{region_size} region (3D: keeping full Z-depth)...")
    start_extract = time.time()
    cropped_image, region_coords = find_spot_dense_region(image, region_size=region_size)
    extract_time = time.time() - start_extract
    print(f"  Cropped image shape (3D): {cropped_image.shape} [Z, Y, X]")
    print(f"  Region coords: Z={region_coords['z_start']}-{region_coords['z_end']}, "
          f"Y={region_coords['y_start']}-{region_coords['y_end']}, "
          f"X={region_coords['x_start']}-{region_coords['x_end']}")
    print(f"  Extract time: {extract_time:.2f}s")
    
    # Save cropped image and metadata
    image_name = Path(image_path).stem
    cropped_path = output_dir / f"{image_name}_region.tif"
    metadata_path = output_dir / f"{image_name}_region_info.json"
    
    print(f"\nSaving cropped image: {cropped_path.name}")
    imwrite(cropped_path, cropped_image)
    
    with open(metadata_path, 'w') as f:
        json.dump({
            'original_image': str(image_path),
            'region_coords': region_coords,
            'cropped_shape': list(cropped_image.shape),
            'thresholds_tested': thresholds,
            'detection_method': 'improved_multi_stage_filtering',
            'parameters': {
                'min_contrast': min_contrast,
                'use_intensity_filter': True,
                'use_size_filter': True
            }
        }, f, indent=2)
    
    # Generate LoG to determine threshold range
    print(f"\nGenerating LoG filter to determine threshold range...")
    from functions.gpu_smfish_v2 import log_filter_gpu
    import torch
    device_torch = torch.device(device if use_gpu else "cpu")
    start_log = time.time()
    log_img = log_filter_gpu(cropped_image, sigma, device=device_torch)
    log_time = time.time() - start_log
    print(f"  LoG image shape (3D): {log_img.shape}")
    print(f"  LoG generation time: {log_time:.2f}s")
    
    # Determine threshold range from LoG values
    log_values = log_img.flatten()
    threshold_min = np.percentile(log_values, 5)   # Bottom 5% (most negative = strongest)
    threshold_max = np.percentile(log_values, 95)  # Top 95%
    
    # Generate threshold values (convert indices 0-16 to actual LoG thresholds)
    threshold_range = threshold_max - threshold_min
    step_size = threshold_range / 16.0
    actual_thresholds = [
        threshold_min + i * step_size
        for i in range(17)
    ]
    
    print(f"  Threshold range: {threshold_min:.2f} to {threshold_max:.2f}")
    print(f"  Testing {len(actual_thresholds)} thresholds")
    
    # Select slice for overlay
    if slice_for_overlay == 'middle':
        z_slice = cropped_image.shape[0] // 2
    else:
        z_slice = slice_for_overlay
    
    # Test each threshold
    print(f"\nTesting {len(thresholds)} thresholds with IMPROVED detection...")
    print(f"  Note: Threshold is optional initial filter - multi-stage filtering will remove background")
    all_results = {}
    detection_times = []
    
    for i, threshold_idx in enumerate(thresholds):
        actual_threshold = actual_thresholds[threshold_idx]
        print(f"  [{i+1}/{len(thresholds)}] Testing threshold_idx={threshold_idx} (LoG={actual_threshold:.2f})...", 
              end=' ', flush=True)
        start_det = time.time()
        
        try:
            # Use improved detection
            # Note: MPS doesn't support max_pool3d, so use CPU for local minima detection
            # The LoG filtering can still use GPU (handled internally)
            detection_device = "cpu" if device == "mps" else (device if use_gpu else "cpu")
            # If min_contrast is 0, skip contrast filter (set to None)
            contrast_thresh = None if min_contrast == 0 else min_contrast
            detected_spots, stats = detect_spots_improved(
                image_np=cropped_image,
                sigma=sigma,
                min_distance=min_distance,
                threshold=actual_threshold,  # Optional initial LoG threshold
                min_contrast=contrast_thresh,  # None = skip contrast filter
                use_intensity_filter=False,  # Disable Raj plateau for now
                min_size_um=0.0,  # No minimum size (spots are very small ~0.15µm)
                max_size_um=10.0,  # More lenient maximum
                device=detection_device,
                return_statistics=True,
            )
            
            det_time = time.time() - start_det
            detection_times.append(det_time)
            print(f"{len(detected_spots)} spots ({det_time:.2f}s)")
            if stats:
                print(f"      Filtering: {stats.get('n_minima', 0)} minima -> "
                      f"{stats.get('n_after_contrast', 0)} after contrast -> "
                      f"{stats.get('n_after_intensity', 0)} after intensity -> "
                      f"{len(detected_spots)} final")
            
            all_results[threshold_idx] = {
                'spots': detected_spots,
                'n_spots': len(detected_spots),
                'detection_time': det_time,
                'actual_threshold': float(actual_threshold),
                'stats': stats
            }
            
            # Create 2D overlay (middle Z-slice)
            spots_in_slice = np.array([]).reshape(0, 2)
            if len(detected_spots) > 0:
                spots_in_slice_full = detected_spots[detected_spots[:, 0] == z_slice]
                if len(spots_in_slice_full) > 0:
                    spots_in_slice = spots_in_slice_full[:, [1, 2]]  # y, x coordinates
            
            overlay_2d_path = output_dir / f"{image_name}_threshold_{threshold_idx:02d}_overlay.tif"
            create_imagej_overlay(
                cropped_image[z_slice],
                spots_in_slice,
                overlay_2d_path,
                spot_color=(1.0, 0.0, 0.0)  # Red spots
            )
            
            # Create 3D overlay (full stack)
            overlay_3d_path = output_3d_dir / f"{image_name}_threshold_{threshold_idx:02d}_3d_overlay.tif"
            create_3d_overlay(
                cropped_image,
                detected_spots,  # All spots in 3D
                overlay_3d_path,
                spot_color=(1.0, 0.0, 0.0)  # Red spots
            )
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results[threshold_idx] = {
                'spots': np.array([]).reshape(0, 3),
                'n_spots': 0,
                'detection_time': time.time() - start_det,
                'actual_threshold': float(actual_threshold),
                'error': str(e)
            }
    
    total_time = time.time() - start_load
    avg_detection_time = np.mean(detection_times) if detection_times else 0
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Summary (IMPROVED Detection)")
    print(f"{'='*60}")
    print(f"Total processing time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"Average detection time per threshold: {avg_detection_time:.2f}s")
    print(f"Total detection time: {sum(detection_times):.2f}s")
    print(f"\nSpot counts by threshold:")
    print(f"{'Thresh Idx':<12} {'LoG Thresh':<15} {'N Spots':<12} {'Time (s)':<12}")
    print(f"{'-'*50}")
    for threshold_idx in sorted(all_results.keys()):
        result = all_results[threshold_idx]
        actual_thresh = result.get('actual_threshold', threshold_idx)
        print(f"{threshold_idx:<12} {actual_thresh:>14.2f}  {result['n_spots']:<12} {result['detection_time']:.2f}")
    
    # Save results summary
    summary_path = output_dir / f"{image_name}_results_summary.json"
    summary = {
        'image_name': image_name,
        'detection_method': 'improved_multi_stage_filtering',
        'timing': {
            'load_time': load_time,
            'extract_time': extract_time,
            'log_time': log_time,
            'total_detection_time': sum(detection_times),
            'avg_detection_time': float(avg_detection_time),
            'total_time': total_time
        },
        'region_coords': region_coords,
        'parameters': {
            'sigma': list(sigma),
            'min_distance': list(min_distance),
            'min_contrast': min_contrast,
            'threshold_range': {
                'min': float(threshold_min),
                'max': float(threshold_max)
            }
        },
        'results': {
            str(k): {
                'n_spots': v['n_spots'],
                'detection_time': v['detection_time'],
                'actual_threshold': v.get('actual_threshold', k),
                'stats': v.get('stats', {})
            }
            for k, v in all_results.items()
        }
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"{'='*60}\n")
    
    return {
        'image_name': image_name,
        'total_time': total_time,
        'avg_detection_time': avg_detection_time,
        'results': all_results
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test thresholds using IMPROVED spot detection (multi-stage filtering)"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--region_size",
        type=int,
        default=1024,
        help="Region size (default: 1024)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml file (default: config.yaml)"
    )
    parser.add_argument(
        "--min_contrast",
        type=float,
        default=1.5,
        help="Minimum local contrast threshold (default: 1.5)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: cuda, cpu, or mps (default: cuda)"
    )
    
    args = parser.parse_args()
    
    test_thresholds_improved(
        args.image_path,
        args.output_dir,
        region_size=args.region_size,
        config_path=args.config,
        min_contrast=args.min_contrast,
        use_gpu=(args.device != "cpu"),
        device=args.device
    )

