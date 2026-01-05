#!/usr/bin/env python3
"""
Find a spot-dense region in an image using Big-FISH detection.
"""

import numpy as np
from tifffile import imread, imwrite
from pathlib import Path
import yaml
import sys

try:
    from bigfish.detection import detect_spots
    HAS_BIGFISH = True
except ImportError:
    HAS_BIGFISH = False
    print("ERROR: Big-FISH is required but not found!")
    sys.exit(1)

try:
    from functions.gpu_smfish_v2 import log_filter_gpu, local_minima_3d_strict
    import torch
    HAS_GPU_FUNCS = True
except ImportError:
    HAS_GPU_FUNCS = False
    print("Warning: GPU functions not available, will use Big-FISH for scanning")


def find_spot_dense_region(image, region_size=1024, threshold=10.25, config=None):
    """
    Find a region with high spot density by doing quick detection on the full image.
    
    Args:
        image: 3D image array (Z, Y, X)
        region_size: Size of region to extract (default: 1024)
        threshold: Threshold for spot detection (default: 10.25)
        config: Config dictionary with parameters
    
    Returns:
        cropped_image, region_coords dict, spot_density_info
    """
    z, y, x = image.shape
    
    print(f"  Scanning full image for spot-dense regions...")
    print(f"  Image shape: {image.shape} [Z, Y, X]")
    
    if HAS_GPU_FUNCS:
        # Use GPU functions for quick scanning (more permissive)
        device = torch.device("cpu")  # Use CPU for this quick scan
        
        # Quick LoG filter
        sigma = tuple(config.get('kernel_size', [1, 1.5, 1.5])) if config else (1, 1.5, 1.5)
        log_img = log_filter_gpu(image, sigma, device=device)
        log_img_neg = -log_img
        
        # Find all local minima (very permissive)
        min_distance = tuple(config.get('minimal_distance', [2, 2, 2])) if config else (2, 2, 2)
        all_coords = local_minima_3d_strict(
            log_img_neg,
            min_distance=min_distance,
            depth_percentile=50.0,  # Keep top 50%
            device=device
        )
        print(f"  Found {len(all_coords)} candidate spots using GPU quick scan")
    else:
        # Fallback to Big-FISH with very permissive threshold
        voxel_size = tuple(config.get('voxel_size', [361, 75, 75])) if config else (361, 75, 75)
        spot_radius = tuple(config.get('spot_size', [600, 300, 300])) if config else (600, 300, 300)
        sigma = tuple(config.get('kernel_size', [1, 1.5, 1.5])) if config else (1, 1.5, 1.5)
        min_distance = tuple(config.get('minimal_distance', [2, 2, 2])) if config else (2, 2, 2)
        
        # Use a very permissive threshold to get many candidates
        permissive_threshold = threshold * 0.5  # Half the threshold for more spots
        all_coords, _ = detect_spots(
            images=image,
            threshold=permissive_threshold,
            return_threshold=True,
            voxel_size=voxel_size,
            spot_radius=spot_radius,
            log_kernel_size=sigma,
            minimum_distance=min_distance,
        )
        print(f"  Found {len(all_coords)} candidate spots using Big-FISH (threshold={permissive_threshold:.2f})")
    
    if len(all_coords) == 0:
        print("  No spots found, using center region")
        # Fallback to center region
        y_start = max(0, (y - region_size) // 2)
        x_start = max(0, (x - region_size) // 2)
        y_end = min(y_start + region_size, y)
        x_end = min(x_start + region_size, x)
        
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
        return cropped, coords, {'method': 'center', 'n_spots': 0}
    
    # Project spots to 2D (sum over Z) to find dense regions
    spot_density = np.zeros((y, x), dtype=np.float32)
    for spot in all_coords:
        if len(spot) == 3:
            z_coord, y_coord, x_coord = int(spot[0]), int(spot[1]), int(spot[2])
        else:
            # Big-FISH returns (y, x) if 2D, but we expect 3D
            continue
        
        if 0 <= y_coord < y and 0 <= x_coord < x:
            spot_density[y_coord, x_coord] += 1
    
    # Smooth the density map
    from scipy.ndimage import gaussian_filter
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
            # Also count actual spots in region
            spots_in_region = np.sum(spot_density[y_start:y_end, x_start:x_end] > 0)
            candidates.append((density_sum, spots_in_region, y_start, y_end, x_start, x_end))
    
    # Sort by density (highest first)
    candidates.sort(reverse=True)
    
    # Pick the best region
    _, n_spots_in_region, y_start, y_end, x_start, x_end = candidates[0]
    
    print(f"  Selected region with density score: {candidates[0][0]:.1f}")
    print(f"  Spots in region: {n_spots_in_region}")
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
    
    density_info = {
        'method': 'spot_density',
        'n_spots_scanned': len(all_coords),
        'n_spots_in_region': n_spots_in_region,
        'density_score': float(candidates[0][0]),
        'top_5_regions': [
            {
                'density_score': float(score),
                'n_spots': int(n_spots),
                'y_range': (int(y_s), int(y_e)),
                'x_range': (int(x_s), int(x_e))
            }
            for score, n_spots, y_s, y_e, x_s, x_e in candidates[:5]
        ]
    }
    
    return cropped, coords, density_info


def main():
    # Image path
    image_path = "/Volumes/Backup Plus/DL_210_data_analysis/Test image/12hr_Amputation_Image2_565.tif"
    config_path = "config.yaml"
    region_size = 1024
    threshold = 10.25
    
    # Load config
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
        print(f"Warning: Config file not found: {config_path}, using defaults")
    
    print("="*60)
    print("Finding Spot-Dense Region")
    print("="*60)
    print(f"Image: {image_path}")
    print(f"Region size: {region_size}x{region_size}")
    print("="*60)
    
    # Load image
    print("\nLoading image...")
    image = imread(image_path)
    print(f"Image shape: {image.shape} [Z, Y, X]")
    
    # Find dense region
    print(f"\nFinding spot-dense region...")
    cropped_image, region_coords, density_info = find_spot_dense_region(
        image,
        region_size=region_size,
        threshold=threshold,
        config=config
    )
    
    print(f"\n{'='*60}")
    print("Region Information")
    print(f"{'='*60}")
    print(f"Region coordinates:")
    print(f"  Z: {region_coords['z_start']}-{region_coords['z_end']} ({region_coords['z_end'] - region_coords['z_start']} slices)")
    print(f"  Y: {region_coords['y_start']}-{region_coords['y_end']} ({region_coords['y_end'] - region_coords['y_start']} pixels)")
    print(f"  X: {region_coords['x_start']}-{region_coords['x_end']} ({region_coords['x_end'] - region_coords['x_start']} pixels)")
    print(f"\nCropped image shape: {cropped_image.shape} [Z, Y, X]")
    print(f"\nDensity information:")
    print(f"  Method: {density_info['method']}")
    print(f"  Spots scanned in full image: {density_info.get('n_spots_scanned', 'N/A')}")
    print(f"  Spots in selected region: {density_info.get('n_spots_in_region', 'N/A')}")
    print(f"  Density score: {density_info.get('density_score', 'N/A'):.1f}")
    
    if 'top_5_regions' in density_info:
        print(f"\nTop 5 candidate regions:")
        for i, region in enumerate(density_info['top_5_regions'], 1):
            print(f"  {i}. Density: {region['density_score']:.1f}, "
                  f"Spots: {region['n_spots']}, "
                  f"Y: {region['y_range'][0]}-{region['y_range'][1]}, "
                  f"X: {region['x_range'][0]}-{region['x_range'][1]}")
    
    # Save cropped region
    image_dir = Path(image_path).parent
    image_name = Path(image_path).stem
    output_path = image_dir / f"{image_name}_dense_region.tif"
    
    print(f"\nSaving dense region to: {output_path}")
    imwrite(output_path, cropped_image)
    
    # Save metadata
    import json
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_json_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_to_json_serializable(item) for item in obj)
        return obj
    
    metadata_path = image_dir / f"{image_name}_dense_region_info.json"
    metadata = {
        'original_image': str(image_path),
        'region_coords': region_coords,
        'cropped_shape': list(cropped_image.shape),
        'density_info': convert_to_json_serializable(density_info),
        'threshold_used': threshold,
        'region_size': region_size
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to: {metadata_path}")
    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")
    
    return cropped_image, region_coords, density_info


if __name__ == "__main__":
    main()

