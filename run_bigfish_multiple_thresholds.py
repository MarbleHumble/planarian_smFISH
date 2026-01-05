#!/usr/bin/env python3
"""
Run Big-FISH spot detection with multiple thresholds (1-16) on an image.
Uses exact threshold values 1, 2, 3, ..., 16.
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
    HAS_BIGFISH = True
except ImportError:
    HAS_BIGFISH = False
    print("ERROR: Big-FISH is required but not found!")
    sys.exit(1)


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


def main():
    # Paths
    image_path = "/Volumes/Backup Plus/DL_210_data_analysis/Test image/12hr_Amputation_Image2_565.tif"
    config_path = "config.yaml"
    thresholds = list(range(1, 17))  # Threshold values: 1, 2, 3, ..., 16
    
    # Load config
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)
    
    # Get parameters from config - use EXACT values from config.yaml
    sigma = tuple(config.get('kernel_size', [1, 1.5, 1.5]))
    min_distance = tuple(config.get('minimal_distance', [2, 2, 2]))
    voxel_size = tuple(config.get('voxel_size', [361, 75, 75]))
    spot_radius = tuple(config.get('spot_size', [600, 300, 300]))
    plot_spot_size = config.get('plot_spot_size', 4)
    plot_inner_circle = config.get('plotInnerCircle', False)
    plot_spot_label = config.get('plotSpotLabel', False)
    
    # Create output directory
    image_dir = Path(image_path).parent
    image_name = Path(image_path).stem
    output_dir = image_dir / f"{image_name}_threshold_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Big-FISH Multiple Threshold Detection")
    print("="*60)
    print(f"Image: {image_path}")
    print(f"Output: {output_dir}")
    print(f"Threshold values to test: {thresholds}")
    print(f"Parameters from config.yaml:")
    print(f"  sigma (kernel_size): {sigma}")
    print(f"  min_distance: {min_distance}")
    print(f"  voxel_size: {voxel_size}")
    print(f"  spot_radius: {spot_radius}")
    print(f"  plot_spot_size: {plot_spot_size}")
    print("="*60)
    
    # Load image
    print("\nLoading image...")
    start_load = time.time()
    image = imread(image_path)
    load_time = time.time() - start_load
    print(f"Image shape: {image.shape} [Z, Y, X]")
    print(f"Load time: {load_time:.2f}s")
    
    # Test each threshold
    print(f"\n{'='*60}")
    print(f"Testing {len(thresholds)} thresholds...")
    print(f"{'='*60}")
    
    results = {}
    all_times = []
    
    for threshold_value in thresholds:
        print(f"\n  Testing threshold={threshold_value}...", end=' ', flush=True)
        
        start_time = time.time()
        detected_spots, threshold_used = detect_spots(
            images=image,
            threshold=threshold_value,
            return_threshold=True,
            voxel_size=voxel_size,
            spot_radius=spot_radius,
            log_kernel_size=sigma,
            minimum_distance=min_distance,
        )
        det_time = time.time() - start_time
        all_times.append(det_time)
        
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
        spot_plot_path = output_dir / f"threshold_{threshold_value:02d}_spotPlot.tif"
        imwrite(spot_plot_path, spot_plot, photometric='minisblack')
        
        # Save spots coordinates
        spots_path = output_dir / f"threshold_{threshold_value:02d}_spots.npy"
        np.save(spots_path, detected_spots)
        
        results[threshold_value] = {
            'n_spots': len(detected_spots),
            'threshold_value': float(threshold_value),
            'threshold_used': float(threshold_used),
            'detection_time': det_time,
            'spot_plot_path': spot_plot_path.name,
            'spots_path': spots_path.name
        }
    
    # Save summary
    summary_path = output_dir / "summary.json"
    summary = {
        'image_path': str(image_path),
        'image_name': image_name,
        'image_shape': list(image.shape),
        'thresholds_tested': thresholds,
        'results': results,
        'parameters': {
            'kernel_size': list(sigma),
            'minimal_distance': list(min_distance),
            'voxel_size': list(voxel_size),
            'spot_size': list(spot_radius),
            'plot_spot_size': plot_spot_size
        },
        'timing': {
            'load_time': load_time,
            'total_detection_time': sum(all_times),
            'avg_detection_time': float(np.mean(all_times)) if all_times else 0.0
        }
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Total detection time: {sum(all_times):.2f}s")
    print(f"Average detection time per threshold: {np.mean(all_times):.2f}s")
    print(f"\nResults by threshold:")
    print(f"{'Threshold':<12} {'N Spots':<12} {'Time (s)':<12}")
    print(f"{'-'*35}")
    for threshold_value in sorted(results.keys()):
        r = results[threshold_value]
        print(f"{threshold_value:<12} {r['n_spots']:<12} {r['detection_time']:.2f}")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Summary saved to: {summary_path.name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
