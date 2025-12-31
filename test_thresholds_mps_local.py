#!/usr/bin/env python3
"""
Test threshold validation on MPS (Mac GPU) - Quick local test.

Extracts 1024x1024 regions from images, tests thresholds 0-16,
and creates ImageJ-compatible overlay visualizations.
"""

import numpy as np
from tifffile import imread, imwrite
from pathlib import Path
import json
import time
import argparse
import yaml
from scipy.spatial.distance import cdist

try:
    from bigfish.detection import detect_spots
    from bigfish.stack import log_filter
    HAS_BIGFISH = True
except ImportError:
    HAS_BIGFISH = False
    print("="*60)
    print("ERROR: Big-FISH is required but not found!")
    print("="*60)
    print("Please activate your conda environment first:")
    print("  conda activate <your_env_name>")
    print("Or use the Python from your environment:")
    print("  /path/to/conda/envs/<env_name>/bin/python test_thresholds_mps_local.py ...")
    print("="*60)
    exit(1)


def extract_random_region(image, region_size=1024):
    """
    Extract a random 1024x1024 region (keeping full Z).
    
    Returns:
        cropped_image, region_coords dict
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


def test_thresholds_on_region(image_path, output_dir, region_size=1024,
                              thresholds=None, config_path=None,
                              sigma=None, min_distance=None, 
                              voxel_size=None, spot_radius=None, 
                              slice_for_overlay='middle'):
    """
    Extract region, test thresholds, and create overlays.
    
    Returns:
        dict with timing and results
    """
    # Load parameters from config.yaml if provided
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        sigma = sigma or tuple(config.get('kernel_size', [1, 1.7, 1.7]))
        min_distance = min_distance or tuple(config.get('minimal_distance', [3, 3, 3]))
        voxel_size = voxel_size or tuple(config.get('voxel_size', [361, 75, 75]))
        spot_radius = spot_radius or tuple(config.get('spot_size', [600, 300, 300]))
        print(f"  Loaded parameters from config.yaml")
        print(f"    sigma (kernel_size): {sigma}")
        print(f"    min_distance: {min_distance}")
        print(f"    voxel_size: {voxel_size}")
        print(f"    spot_radius (spot_size): {spot_radius}")
    else:
        # Default values (from config.yaml defaults)
        sigma = sigma or (1, 1.7, 1.7)
        min_distance = min_distance or (3, 3, 3)
        voxel_size = voxel_size or (361, 75, 75)
        spot_radius = spot_radius or (600, 300, 300)
    
    if thresholds is None:
        thresholds = list(range(17))  # 0-16
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Processing: {Path(image_path).name}")
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
    cropped_image, region_coords = extract_random_region(image, region_size=region_size)
    extract_time = time.time() - start_extract
    print(f"  Cropped image shape (3D): {cropped_image.shape} [Z, Y, X]")
    print(f"  Region coords: Z={region_coords['z_start']}-{region_coords['z_end']}, "
          f"Y={region_coords['y_start']}-{region_coords['y_end']}, "
          f"X={region_coords['x_start']}-{region_coords['x_end']}")
    print(f"  Extract time: {extract_time:.2f}s")
    
    # Save cropped image and metadata
    image_name = Path(image_path).stem
    
    # Create separate folder for 3D overlays
    output_3d_dir = output_dir / "3D_overlays"
    output_3d_dir.mkdir(parents=True, exist_ok=True)
    
    cropped_path = output_dir / f"{image_name}_region.tif"
    metadata_path = output_dir / f"{image_name}_region_info.json"
    
    print(f"\nSaving cropped image: {cropped_path.name}")
    imwrite(cropped_path, cropped_image)
    
    with open(metadata_path, 'w') as f:
        json.dump({
            'original_image': str(image_path),
            'region_coords': region_coords,
            'cropped_shape': list(cropped_image.shape),
            'thresholds_tested': thresholds
        }, f, indent=2)
    
    # Generate LoG (once, reuse for all thresholds) - 3D processing
    print(f"\nGenerating 3D LoG filter (sigma={sigma})...")
    start_log = time.time()
    log_img = log_filter(cropped_image, sigma)
    log_time = time.time() - start_log
    print(f"  LoG image shape (3D): {log_img.shape}")
    print(f"  LoG generation time: {log_time:.2f}s")
    
    # Select slice for overlay
    if slice_for_overlay == 'middle':
        z_slice = cropped_image.shape[0] // 2
    else:
        z_slice = slice_for_overlay
    
    # Test each threshold
    print(f"\nTesting {len(thresholds)} thresholds...")
    all_results = {}
    detection_times = []
    
    for i, threshold in enumerate(thresholds):
        print(f"  [{i+1}/{len(thresholds)}] Testing threshold={threshold}...", end=' ', flush=True)
        start_det = time.time()
        
        try:
            # 3D spot detection using Big-FISH
            detected_spots, _ = detect_spots(
                images=cropped_image,
                threshold=threshold,
                return_threshold=True,
                voxel_size=voxel_size,
                spot_radius=spot_radius,
                log_kernel_size=sigma,
                minimum_distance=min_distance,
            )
            # detected_spots shape: [N_spots, 3] with coordinates [z, y, x]
            
            det_time = time.time() - start_det
            detection_times.append(det_time)
            print(f"{len(detected_spots)} spots ({det_time:.2f}s)")
            
            all_results[threshold] = {
                'spots': detected_spots,
                'n_spots': len(detected_spots),
                'detection_time': det_time
            }
            
            # Create overlay for this threshold (middle Z-slice)
            # Always create overlay, even if no spots (shows background image)
            spots_in_slice = np.array([]).reshape(0, 2)
            if len(detected_spots) > 0:
                # Get spots in this Z-slice
                spots_in_slice = detected_spots[detected_spots[:, 0] == z_slice]
                if len(spots_in_slice) > 0:
                    # Convert to (y, x) for 2D overlay
                    spots_in_slice = spots_in_slice[:, [1, 2]]  # y, x coordinates
            
            # Create 2D overlay (middle Z-slice) - will be empty if no spots in this slice
            overlay_2d_path = output_dir / f"{image_name}_threshold_{threshold:02d}_overlay.tif"
            create_imagej_overlay(
                cropped_image[z_slice],
                spots_in_slice,
                overlay_2d_path,
                spot_color=(1.0, 0.0, 0.0)  # Red spots
            )
            
            # Create 3D overlay (full stack)
            overlay_3d_path = output_3d_dir / f"{image_name}_threshold_{threshold:02d}_3d_overlay.tif"
            create_3d_overlay(
                cropped_image,
                detected_spots,  # All spots in 3D
                overlay_3d_path,
                spot_color=(1.0, 0.0, 0.0)  # Red spots
            )
            
        except Exception as e:
            print(f"ERROR: {e}")
            all_results[threshold] = {
                'spots': np.array([]).reshape(0, 3),
                'n_spots': 0,
                'detection_time': time.time() - start_det,
                'error': str(e)
            }
    
    total_time = time.time() - start_load
    avg_detection_time = np.mean(detection_times) if detection_times else 0
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"Total processing time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"Average detection time per threshold: {avg_detection_time:.2f}s")
    print(f"Total detection time: {sum(detection_times):.2f}s")
    print(f"\nSpot counts by threshold:")
    print(f"{'Threshold':<12} {'N Spots':<12} {'Time (s)':<12}")
    print(f"{'-'*36}")
    for threshold in sorted(all_results.keys()):
        result = all_results[threshold]
        print(f"{threshold:<12} {result['n_spots']:<12} {result['detection_time']:.2f}")
    
    # Save results summary
    summary_path = output_dir / f"{image_name}_results_summary.json"
    summary = {
        'image_name': image_name,
        'timing': {
            'load_time': load_time,
            'extract_time': extract_time,
            'log_time': log_time,
            'total_detection_time': sum(detection_times),
            'avg_detection_time': float(avg_detection_time),
            'total_time': total_time
        },
        'region_coords': region_coords,
        'results': {
            str(k): {
                'n_spots': v['n_spots'],
                'detection_time': v['detection_time']
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


def process_dataset(data_dir, output_base_dir, region_size=1024,
                   thresholds=None, config_path=None, test_one_image=False):
    """
    Process all images in the dataset structure.
    """
    data_dir = Path(data_dir)
    output_base_dir = Path(output_base_dir)
    
    if thresholds is None:
        thresholds = list(range(17))  # 0-16
    
    # Find all 565 channel images
    print(f"Scanning for 565 channel images in: {data_dir}")
    image_files = []
    
    for condition_dir in sorted(data_dir.iterdir()):
        if not condition_dir.is_dir() or condition_dir.name.startswith('.'):
            continue
        
        print(f"  Checking condition: {condition_dir.name}")
        for image_dir in sorted(condition_dir.glob('Image*')):
            if not image_dir.is_dir():
                continue
            
            channel_565_dir = image_dir / '565'
            if channel_565_dir.exists():
                # Get only files directly in 565 directory (maxdepth 1 equivalent)
                tif_files = [f for f in channel_565_dir.glob('*.tif') 
                            if f.parent == channel_565_dir and not f.name.startswith('._')]
                
                print(f"    Found {len(tif_files)} .tif files in {image_dir.name}/565/")
                for tif_file in tif_files:
                    image_files.append({
                        'path': tif_file,
                        'condition': condition_dir.name,
                        'image_name': image_dir.name,
                        'full_name': f"{condition_dir.name}_{image_dir.name}"
                    })
    
    print(f"Found {len(image_files)} images")
    
    if len(image_files) == 0:
        print("No images found!")
        return
    
    if test_one_image:
        print(f"\n*** TESTING ONE IMAGE ONLY ***")
        image_files = image_files[:1]
    
    # Process each image
    all_timings = []
    
    for i, img_info in enumerate(image_files):
        print(f"\n{'#'*60}")
        print(f"Image {i+1}/{len(image_files)}: {img_info['full_name']}")
        print(f"{'#'*60}")
        
        # Create output directory for this image
        output_dir = output_base_dir / img_info['full_name']
        
        try:
            result = test_thresholds_on_region(
                img_info['path'],
                output_dir,
                region_size=region_size,
                thresholds=thresholds,
                config_path=config_path
            )
            all_timings.append(result['total_time'])
            
        except Exception as e:
            print(f"ERROR processing {img_info['full_name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final summary
    if len(all_timings) > 0:
        print(f"\n{'='*60}")
        print(f"FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"Processed {len(all_timings)} images")
        print(f"Average time per image: {np.mean(all_timings):.2f}s ({np.mean(all_timings)/60:.2f} minutes)")
        print(f"Total time: {sum(all_timings):.2f}s ({sum(all_timings)/60:.2f} minutes)")
        print(f"Estimated time for all {len(image_files)} images: "
              f"{np.mean(all_timings) * len(image_files):.2f}s "
              f"({np.mean(all_timings) * len(image_files) / 60:.2f} minutes)")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test threshold validation on MPS - Extract regions and create overlays"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/Volumes/Backup Plus/Experiment_results/306_analysis_results/Experiment",
        help="Base directory containing condition/Image*/565/*.tif files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/Users/eliasguan/Desktop/DL_210_data_analysis/ML_data_set_trainning/Test thresholding",
        help="Output directory"
    )
    parser.add_argument(
        "--region_size",
        type=int,
        default=1024,
        help="Region size (default: 1024)"
    )
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs='+',
        default=None,
        help="Thresholds to test (default: 0-16)"
    )
    parser.add_argument(
        "--test_one",
        action="store_true",
        help="Test on one image only"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml file (default: config.yaml)"
    )
    
    args = parser.parse_args()
    
    process_dataset(
        args.data_dir,
        args.output_dir,
        region_size=args.region_size,
        thresholds=args.thresholds,
        config_path=args.config,
        test_one_image=args.test_one
    )

