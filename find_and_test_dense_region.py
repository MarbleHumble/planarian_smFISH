#!/usr/bin/env python3
"""
Find a random dense region in a full image and test Big-FISH with multiple thresholds.
"""

import numpy as np
from tifffile import imread, imwrite
from pathlib import Path
import yaml
import sys
import json
import time

try:
    from bigfish.detection import detect_spots
    from bigfish.stack import log_filter
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


def generate_coordinates_2D(y, x, shape, iteration=4, get_inner_spot=False):
    """Generate coordinates for 2D spot plotting."""
    coordinates_collection = [(y, x)]
    max_y = shape[0] - 1
    max_x = shape[1] - 1

    for _ in range(iteration):
        current_coordinates = coordinates_collection.copy()
        for coord in current_coordinates:
            cy, cx = coord
            if cx + 1 <= max_x:
                coordinates_collection.append((cy, cx + 1))
            if 0 <= cx - 1:
                coordinates_collection.append((cy, cx - 1))
            if cy + 1 <= max_y:
                coordinates_collection.append((cy + 1, cx))
            if 0 <= cy - 1:
                coordinates_collection.append((cy - 1, cx))
    
    coordinates_collection = list(set(coordinates_collection))
    
    if get_inner_spot:
        return coordinates_collection
    else:
        coordinates_collection = [
            coord for coord in coordinates_collection 
            if abs(coord[0] - y) + abs(coord[1] - x) == iteration
        ]
        return coordinates_collection


def create_spot_plot(image, spots, plot_spot_size=4, plot_inner_circle=False, plot_spot_label=False):
    """Create a 3D spot plot from detected spots."""
    z, y, x = image.shape
    
    if plot_spot_label:
        spot_plot = np.zeros(image.shape, dtype=np.uint32)
    else:
        spot_plot = np.zeros(image.shape, dtype=np.uint8)
    
    shape_2d = [spot_plot.shape[1], spot_plot.shape[2]]
    
    for i, spot in enumerate(spots):
        z_coord = int(spot[0])
        y_coord = int(spot[1])
        x_coord = int(spot[2])
        
        plot_locations = generate_coordinates_2D(
            y_coord, x_coord, 
            shape_2d, 
            iteration=plot_spot_size,
            get_inner_spot=plot_inner_circle
        )
        
        for plot_y, plot_x in plot_locations:
            if 0 <= z_coord < z and 0 <= plot_y < y and 0 <= plot_x < x:
                if plot_spot_label:
                    spot_plot[z_coord, plot_y, plot_x] = i + 1
                else:
                    spot_plot[z_coord, plot_y, plot_x] = 255
    
    return spot_plot


def find_spot_dense_region(image, region_size=1024, config=None):
    """Find a region with high spot density."""
    z, y, x = image.shape
    
    print(f"  Scanning full image for spot-dense regions...")
    print(f"  Full image shape: {image.shape} [Z, Y, X]")
    
    if HAS_GPU_FUNCS and y > region_size and x > region_size:
        # Use GPU functions for quick scanning (more permissive)
        device = torch.device("cpu")
        sigma = tuple(config.get('kernel_size', [1, 1.5, 1.5])) if config else (1, 1.5, 1.5)
        log_img = log_filter_gpu(image, sigma, device=device)
        log_img_neg = -log_img
        
        min_distance = tuple(config.get('minimal_distance', [2, 2, 2])) if config else (2, 2, 2)
        all_coords = local_minima_3d_strict(
            log_img_neg,
            min_distance=min_distance,
            depth_percentile=50.0,
            device=device
        )
        print(f"  Found {len(all_coords)} candidate spots using GPU quick scan")
    else:
        # If image is already small or GPU not available, use center or random
        print(f"  Image is {region_size}x{region_size} or smaller, using center region")
        all_coords = []
    
    if len(all_coords) == 0 or (y <= region_size and x <= region_size):
        # Use center region if image is small or no spots found
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
    
    # Project spots to 2D
    spot_density = np.zeros((y, x), dtype=np.float32)
    for spot in all_coords:
        if len(spot) == 3:
            z_coord, y_coord, x_coord = int(spot[0]), int(spot[1]), int(spot[2])
            if 0 <= y_coord < y and 0 <= x_coord < x:
                spot_density[y_coord, x_coord] += 1
    
    # Smooth the density map
    from scipy.ndimage import gaussian_filter
    spot_density_smooth = gaussian_filter(spot_density, sigma=region_size//4)
    
    # Find candidate regions
    step = region_size // 2
    candidates = []
    
    for y_start in range(0, y - region_size + 1, step):
        for x_start in range(0, x - region_size + 1, step):
            y_end = min(y_start + region_size, y)
            x_end = min(x_start + region_size, x)
            
            density_sum = spot_density_smooth[y_start:y_end, x_start:x_end].sum()
            spots_in_region = np.sum(spot_density[y_start:y_end, x_start:x_end] > 0)
            candidates.append((density_sum, spots_in_region, y_start, y_end, x_start, x_end))
    
    # Sort by density and pick one of the top candidates randomly
    candidates.sort(reverse=True)
    
    # Pick randomly from top 5 candidates for some variety
    import random
    top_n = min(5, len(candidates))
    selected = random.choice(candidates[:top_n])
    
    _, n_spots_in_region, y_start, y_end, x_start, x_end = selected
    
    print(f"  Selected region (random from top {top_n}):")
    print(f"    Density score: {selected[0]:.1f}")
    print(f"    Spots in region: {n_spots_in_region}")
    print(f"    Region: Y={y_start}-{y_end}, X={x_start}-{x_end}")
    
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
    
    return cropped, coords, {
        'method': 'spot_density_random',
        'n_spots_scanned': len(all_coords),
        'n_spots_in_region': int(n_spots_in_region),
        'density_score': float(selected[0])
    }


def test_multiple_thresholds(image, region_coords, thresholds, config, output_dir):
    """Test Big-FISH detection with multiple thresholds."""
    sigma = tuple(config.get('kernel_size', [1, 1.5, 1.5]))
    min_distance = tuple(config.get('minimal_distance', [2, 2, 2]))
    voxel_size = tuple(config.get('voxel_size', [361, 75, 75]))
    spot_radius = tuple(config.get('spot_size', [600, 300, 300]))
    plot_spot_size = config.get('plot_spot_size', 4)
    plot_inner_circle = config.get('plotInnerCircle', False)
    plot_spot_label = config.get('plotSpotLabel', False)
    
    # Generate LoG to determine threshold range
    print(f"\nGenerating LoG filter to determine threshold range...")
    log_img = log_filter(image, sigma)
    
    log_values = log_img.flatten()
    threshold_min = np.percentile(log_values, 5)
    threshold_max = np.percentile(log_values, 95)
    threshold_range = threshold_max - threshold_min
    step_size = threshold_range / 16.0
    
    actual_thresholds = [
        threshold_min + i * step_size
        for i in range(17)
    ]
    
    print(f"  Threshold range: {threshold_min:.2f} to {threshold_max:.2f}")
    
    results = {}
    
    # Create 3D overlays directory
    overlay_3d_dir = output_dir / "3D_overlays"
    overlay_3d_dir.mkdir(exist_ok=True)
    
    for threshold_idx in thresholds:
        if threshold_idx < 0 or threshold_idx >= len(actual_thresholds):
            print(f"  Skipping invalid threshold index: {threshold_idx}")
            continue
        
        actual_threshold = actual_thresholds[threshold_idx]
        print(f"\n  Testing threshold index {threshold_idx} (LoG={actual_threshold:.2f})...", end=' ', flush=True)
        
        start_time = time.time()
        detected_spots, threshold_used = detect_spots(
            images=image,
            threshold=actual_threshold,
            return_threshold=True,
            voxel_size=voxel_size,
            spot_radius=spot_radius,
            log_kernel_size=sigma,
            minimum_distance=min_distance,
        )
        det_time = time.time() - start_time
        
        print(f"{len(detected_spots)} spots ({det_time:.2f}s)")
        
        # Create spot plot
        spot_plot = create_spot_plot(
            image,
            detected_spots,
            plot_spot_size=plot_spot_size,
            plot_inner_circle=plot_inner_circle,
            plot_spot_label=plot_spot_label
        )
        
        # Save spot plot
        spot_plot_path = output_dir / f"threshold_{threshold_idx:02d}_spotPlot.tif"
        imwrite(spot_plot_path, spot_plot, photometric='minisblack')
        
        # Save spots coordinates
        spots_path = output_dir / f"threshold_{threshold_idx:02d}_spots.npy"
        np.save(spots_path, detected_spots)
        
        results[threshold_idx] = {
            'n_spots': len(detected_spots),
            'threshold_value': float(actual_threshold),
            'threshold_used': float(threshold_used),
            'detection_time': det_time,
            'spot_plot_path': str(spot_plot_path.name),
            'spots_path': str(spots_path.name)
        }
    
    return results, actual_thresholds


def main():
    # Paths
    image_path = "/Volumes/Backup Plus/Experiment_results/306_analysis_results/Experiment/12hr_Amputation/Image2/565/12hr_Amputation_Image2_565.tif"
    output_base = "/Volumes/Backup Plus/DL_210_data_analysis/Test image"
    config_path = "config.yaml"
    region_size = 1024
    thresholds = list(range(1, 17))  # 1-16
    
    # Load config
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
        print(f"Warning: Config file not found: {config_path}, using defaults")
    
    # Create output directory
    image_name = Path(image_path).stem
    output_dir = Path(output_base) / f"{image_name}_dense_region_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Find Dense Region and Test Multiple Thresholds")
    print("="*60)
    print(f"Image: {image_path}")
    print(f"Output: {output_dir}")
    print(f"Region size: {region_size}x{region_size}")
    print(f"Thresholds to test: {thresholds}")
    print("="*60)
    
    # Load image
    print("\nLoading image...")
    start_load = time.time()
    image = imread(image_path)
    load_time = time.time() - start_load
    print(f"Image shape: {image.shape} [Z, Y, X]")
    print(f"Load time: {load_time:.2f}s")
    
    # Find dense region
    print(f"\nFinding dense region...")
    cropped_image, region_coords, density_info = find_spot_dense_region(
        image,
        region_size=region_size,
        config=config
    )
    
    print(f"\nRegion coordinates:")
    print(f"  Z: {region_coords['z_start']}-{region_coords['z_end']}")
    print(f"  Y: {region_coords['y_start']}-{region_coords['y_end']}")
    print(f"  X: {region_coords['x_start']}-{region_coords['x_end']}")
    print(f"  Cropped shape: {cropped_image.shape} [Z, Y, X]")
    
    # Save cropped region
    region_path = output_dir / f"{image_name}_dense_region.tif"
    print(f"\nSaving dense region to: {region_path.name}")
    imwrite(region_path, cropped_image)
    
    # Test multiple thresholds
    print(f"\n{'='*60}")
    print("Testing multiple thresholds...")
    print(f"{'='*60}")
    results, actual_thresholds = test_multiple_thresholds(
        cropped_image,
        region_coords,
        thresholds,
        config,
        output_dir
    )
    
    # Save summary
    summary_path = output_dir / "summary.json"
    summary = {
        'image_path': str(image_path),
        'image_name': image_name,
        'region_coords': region_coords,
        'region_shape': list(cropped_image.shape),
        'density_info': {
            'method': density_info['method'],
            'n_spots_scanned': density_info.get('n_spots_scanned', 0),
            'n_spots_in_region': density_info.get('n_spots_in_region', 0),
            'density_score': density_info.get('density_score', 0.0)
        },
        'threshold_range': {
            'min': float(np.min(actual_thresholds)),
            'max': float(np.max(actual_thresholds)),
            'actual_thresholds': [float(t) for t in actual_thresholds]
        },
        'results': results,
        'parameters': {
            'kernel_size': list(config.get('kernel_size', [1, 1.5, 1.5])),
            'minimal_distance': list(config.get('minimal_distance', [2, 2, 2])),
            'voxel_size': list(config.get('voxel_size', [361, 75, 75])),
            'spot_size': list(config.get('spot_size', [600, 300, 300])),
            'plot_spot_size': config.get('plot_spot_size', 4)
        }
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Region: Y={region_coords['y_start']}-{region_coords['y_end']}, "
          f"X={region_coords['x_start']}-{region_coords['x_end']}")
    print(f"\nResults by threshold:")
    print(f"{'Thresh Idx':<12} {'LoG Thresh':<15} {'N Spots':<12} {'Time (s)':<12}")
    print(f"{'-'*50}")
    for idx in sorted(results.keys()):
        r = results[idx]
        print(f"{idx:<12} {r['threshold_value']:>14.2f}  {r['n_spots']:<12} {r['detection_time']:.2f}")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Summary saved to: {summary_path.name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

