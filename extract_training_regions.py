#!/usr/bin/env python3
"""
extract_training_regions.py
Extract random 1024x1024 regions from 3D images for ML training.

This script:
1. Loads 3D images and corresponding spot annotations
2. Extracts random 1024x1024 regions (keeping full Z dimension)
3. Crops annotations to match regions
4. Crops spot plots for validation
5. Saves everything in organized structure
"""

import numpy as np
from tifffile import imread, imwrite
from pathlib import Path
import json
import argparse
from typing import Tuple, List, Dict


def extract_region_from_3d(image: np.ndarray, 
                           y_start: int, y_end: int,
                           x_start: int, x_end: int) -> np.ndarray:
    """
    Extract a region from 3D image (keep full Z dimension).
    
    Args:
        image: 3D image array (Z, Y, X)
        y_start, y_end: Y coordinates (inclusive start, exclusive end)
        x_start, x_end: X coordinates (inclusive start, exclusive end)
    
    Returns:
        cropped_region: Cropped image (Z, y_end-y_start, x_end-x_start)
    """
    return image[:, y_start:y_end, x_start:x_end]


def crop_spots_to_region(spots: np.ndarray, 
                         y_start: int, y_end: int,
                         x_start: int, x_end: int,
                         z_start: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop spot coordinates to match image region.
    
    Args:
        spots: Spot coordinates array (N, 3) where columns are [Z, Y, X]
        y_start, y_end: Y coordinates of region
        x_start, x_end: X coordinates of region
        z_start: Z start coordinate (default 0, keeping full Z stack)
    
    Returns:
        cropped_spots: Spots within region (coordinates relative to region)
        valid_mask: Boolean array indicating which spots are in region
    """
    if len(spots) == 0:
        return np.array([]).reshape(0, 3), np.array([], dtype=bool)
    
    # Filter spots within region (keep full Z dimension)
    valid_mask = (
        (spots[:, 0] >= z_start) &  # Z coordinate (keeping full stack)
        (spots[:, 1] >= y_start) & (spots[:, 1] < y_end) &  # Y coordinate
        (spots[:, 2] >= x_start) & (spots[:, 2] < x_end)    # X coordinate
    )
    
    cropped_spots = spots[valid_mask].copy()
    
    # Adjust coordinates to be relative to region origin
    cropped_spots[:, 0] -= z_start  # Z offset
    cropped_spots[:, 1] -= y_start  # Y offset
    cropped_spots[:, 2] -= x_start  # X offset
    
    return cropped_spots, valid_mask


def get_random_region_coords(image_shape: Tuple[int, int, int],
                             region_size: Tuple[int, int] = (1024, 1024),
                             random_seed: int = None) -> Dict[str, int]:
    """
    Get random region coordinates for 3D image.
    
    Args:
        image_shape: (Z, Y, X) shape of image
        region_size: (height, width) of region to extract
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary with region coordinates
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    Z, Y, X = image_shape
    region_h, region_w = region_size
    
    # Random start positions (keeping full Z dimension)
    y_start = np.random.randint(0, max(1, Y - region_h + 1))
    x_start = np.random.randint(0, max(1, X - region_w + 1))
    z_start = 0  # Keep full Z stack
    
    y_end = min(y_start + region_h, Y)
    x_end = min(x_start + region_w, X)
    z_end = Z  # Keep full Z stack
    
    return {
        'z_start': z_start,
        'z_end': z_end,
        'y_start': y_start,
        'y_end': y_end,
        'x_start': x_start,
        'x_end': x_end,
        'z_size': z_end - z_start,
        'y_size': y_end - y_start,
        'x_size': x_end - x_start,
    }


def extract_regions_from_dataset(data_dir: Path,
                                  output_dir: Path,
                                  region_size: Tuple[int, int] = (1024, 1024),
                                  n_regions_per_image: int = 1,
                                  threshold: float = 10.25,
                                  random_seed: int = 42):
    """
    Extract random regions from all images in dataset.
    
    Args:
        data_dir: Root directory containing:
            - Original Images/ (or original_image/)
            - annotations/
            - Spot_plots/ (or original_images/)
        output_dir: Output directory for extracted regions
        region_size: (height, width) of regions to extract
        n_regions_per_image: Number of random regions per image
        threshold: Threshold value (saved in metadata)
        random_seed: Random seed for reproducibility
    """
    # Find input directories
    original_dir = None
    for possible_name in ['Original Images', 'original_image', 'original_images']:
        possible_path = data_dir / possible_name
        if possible_path.exists():
            original_dir = possible_path
            break
    
    if original_dir is None:
        raise FileNotFoundError(f"Could not find original images directory in {data_dir}")
    
    annotations_dir = data_dir / 'annotations'
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")
    
    spot_plots_dir = None
    for possible_name in ['Spot_plots', 'original_images', 'spot_plots']:
        possible_path = data_dir / possible_name
        if possible_path.exists():
            spot_plots_dir = possible_path
            break
    
    # Create output directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    regions_dir = output_dir / 'regions'
    regions_dir.mkdir(exist_ok=True)
    
    # Process each image
    image_files = sorted(list(original_dir.glob('*.tif')))
    
    if len(image_files) == 0:
        raise ValueError(f"No .tif files found in {original_dir}")
    
    print(f"Found {len(image_files)} images")
    print(f"Extracting {n_regions_per_image} region(s) per image")
    print(f"Region size: {region_size[0]}x{region_size[1]} pixels (keeping full Z stack)\n")
    
    all_metadata = []
    global_region_idx = 0
    
    for img_file in image_files:
        image_name = img_file.stem
        print(f"Processing: {image_name}")
        
        # Find corresponding annotation file
        annotation_file = annotations_dir / f"{image_name}_spots.npy"
        if not annotation_file.exists():
            print(f"  Warning: No annotation file found ({annotation_file.name}), skipping")
            continue
        
        # Find corresponding spot plot
        spot_plot_file = None
        if spot_plots_dir:
            for pattern in [f"{image_name}_plot.tif", f"{image_name}.tif"]:
                possible_file = spot_plots_dir / pattern
                if possible_file.exists():
                    spot_plot_file = possible_file
                    break
        
        # Load data
        print(f"  Loading image...")
        image = imread(img_file)
        print(f"    Image shape: {image.shape} (Z, Y, X)")
        
        print(f"  Loading annotations...")
        spots = np.load(annotation_file)
        print(f"    Found {len(spots)} spots")
        
        spot_plot = None
        if spot_plot_file:
            print(f"  Loading spot plot...")
            spot_plot = imread(spot_plot_file)
            print(f"    Spot plot shape: {spot_plot.shape}")
        
        # Extract regions
        for region_idx in range(n_regions_per_image):
            global_region_idx += 1
            region_name = f"{image_name}_region{region_idx+1:02d}"
            region_output_dir = regions_dir / region_name
            region_output_dir.mkdir(exist_ok=True)
            
            print(f"\n  Extracting region {region_idx+1}/{n_regions_per_image}: {region_name}")
            
            # Get random region coordinates
            region_coords = get_random_region_coords(
                image.shape,
                region_size=region_size,
                random_seed=random_seed + global_region_idx  # Different seed for each region
            )
            
            print(f"    Region coordinates:")
            print(f"      Z: {region_coords['z_start']}-{region_coords['z_end']} ({region_coords['z_size']} slices)")
            print(f"      Y: {region_coords['y_start']}-{region_coords['y_end']} ({region_coords['y_size']} pixels)")
            print(f"      X: {region_coords['x_start']}-{region_coords['x_end']} ({region_coords['x_size']} pixels)")
            
            # Extract image region
            image_region = extract_region_from_3d(
                image,
                region_coords['y_start'], region_coords['y_end'],
                region_coords['x_start'], region_coords['x_end']
            )
            
            # Crop spots to region
            spots_region, valid_mask = crop_spots_to_region(
                spots,
                region_coords['y_start'], region_coords['y_end'],
                region_coords['x_start'], region_coords['x_end'],
                z_start=region_coords['z_start']
            )
            
            print(f"    Spots in region: {len(spots_region)} / {len(spots)}")
            
            # Crop spot plot if available
            spot_plot_region = None
            if spot_plot is not None:
                if spot_plot.ndim == 3 and spot_plot.shape == image.shape:
                    # Same dimensions, crop the same way
                    spot_plot_region = extract_region_from_3d(
                        spot_plot,
                        region_coords['y_start'], region_coords['y_end'],
                        region_coords['x_start'], region_coords['x_end']
                    )
                elif spot_plot.ndim == 2:
                    # 2D plot, crop Y and X only
                    spot_plot_region = spot_plot[
                        region_coords['y_start']:region_coords['y_end'],
                        region_coords['x_start']:region_coords['x_end']
                    ]
            
            # Save extracted data
            print(f"    Saving region data...")
            
            # Save image region
            image_region_file = region_output_dir / 'image_region.tif'
            imwrite(image_region_file, image_region)
            print(f"      Saved: {image_region_file.name}")
            
            # Save spots
            spots_region_file = region_output_dir / 'spots.npy'
            np.save(spots_region_file, spots_region)
            print(f"      Saved: {spots_region_file.name} ({len(spots_region)} spots)")
            
            # Save spot plot region
            if spot_plot_region is not None:
                spot_plot_file = region_output_dir / 'spot_plot_region.tif'
                imwrite(spot_plot_file, spot_plot_region)
                print(f"      Saved: {spot_plot_file.name}")
            
            # Save metadata
            metadata = {
                'region_name': region_name,
                'source_image': image_name,
                'threshold': threshold,
                'region_coords': region_coords,
                'original_image_shape': list(image.shape),
                'region_shape': list(image_region.shape),
                'n_spots_original': len(spots),
                'n_spots_in_region': len(spots_region),
                'spot_coverage': float(len(spots_region) / len(spots)) if len(spots) > 0 else 0.0,
            }
            
            metadata_file = region_output_dir / 'region_info.json'
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"      Saved: {metadata_file.name}")
            
            all_metadata.append(metadata)
    
    # Save global metadata
    global_metadata = {
        'dataset_info': {
            'n_images': len(image_files),
            'n_regions_total': len(all_metadata),
            'n_regions_per_image': n_regions_per_image,
            'region_size': region_size,
            'threshold': threshold,
        },
        'regions': all_metadata
    }
    
    metadata_summary_file = output_dir / 'dataset_metadata.json'
    with open(metadata_summary_file, 'w') as f:
        json.dump(global_metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Region extraction complete!")
    print(f"{'='*60}")
    print(f"Total regions extracted: {len(all_metadata)}")
    print(f"Output directory: {output_dir}")
    print(f"Metadata saved to: {metadata_summary_file}")
    print(f"\nRegion structure:")
    print(f"  {regions_dir}/")
    print(f"    region_name/")
    print(f"      image_region.tif")
    print(f"      spots.npy")
    print(f"      spot_plot_region.tif")
    print(f"      region_info.json")


def main():
    parser = argparse.ArgumentParser(
        description="Extract random regions from 3D images for ML training"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory containing Original Images/, annotations/, and Spot_plots/"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for extracted regions"
    )
    parser.add_argument(
        "--region_size",
        type=int,
        nargs=2,
        default=[1024, 1024],
        metavar=('HEIGHT', 'WIDTH'),
        help="Size of regions to extract (default: 1024 1024)"
    )
    parser.add_argument(
        "--n_regions",
        type=int,
        default=1,
        help="Number of random regions to extract per image (default: 1)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=10.25,
        help="Threshold value to record in metadata (default: 10.25)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    extract_regions_from_dataset(
        data_dir=data_dir,
        output_dir=output_dir,
        region_size=tuple(args.region_size),
        n_regions_per_image=args.n_regions,
        threshold=args.threshold,
        random_seed=args.seed
    )


if __name__ == "__main__":
    main()

