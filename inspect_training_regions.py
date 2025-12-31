#!/usr/bin/env python3
"""
Quick script to inspect extracted training regions.
"""

import numpy as np
from tifffile import imread
from pathlib import Path
import json

def inspect_regions(regions_dir):
    """Inspect extracted regions."""
    regions_dir = Path(regions_dir)
    
    if not regions_dir.exists():
        print(f"Error: Directory not found: {regions_dir}")
        return
    
    region_dirs = sorted([d for d in regions_dir.iterdir() if d.is_dir()])
    
    if len(region_dirs) == 0:
        print(f"No region directories found in {regions_dir}")
        return
    
    print("="*70)
    print("TRAINING REGIONS INSPECTION")
    print("="*70)
    print(f"\nFound {len(region_dirs)} regions\n")
    
    for region_dir in region_dirs:
        print(f"\n{'='*70}")
        print(f"Region: {region_dir.name}")
        print(f"{'='*70}")
        
        # Check files
        image_file = region_dir / 'image_region.tif'
        spots_file = region_dir / 'spots.npy'
        spot_plot_file = region_dir / 'spot_plot_region.tif'
        metadata_file = region_dir / 'region_info.json'
        
        # Load metadata
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
            
            print(f"\nMetadata:")
            print(f"  Source image: {metadata['source_image']}")
            print(f"  Threshold: {metadata['threshold']}")
            print(f"  Original image shape: {metadata['original_image_shape']} (Z, Y, X)")
            print(f"  Region shape: {metadata['region_shape']} (Z, Y, X)")
            print(f"  Region coordinates:")
            coords = metadata['region_coords']
            print(f"    Z: {coords['z_start']}-{coords['z_end']} ({coords['z_size']} slices)")
            print(f"    Y: {coords['y_start']}-{coords['y_end']} ({coords['y_size']} pixels)")
            print(f"    X: {coords['x_start']}-{coords['x_end']} ({coords['x_size']} pixels)")
            print(f"  Spots: {metadata['n_spots_in_region']} / {metadata['n_spots_original']} ({metadata['spot_coverage']*100:.1f}% coverage)")
        
        # Load and inspect image
        if image_file.exists():
            print(f"\nImage region:")
            image = imread(image_file)
            print(f"  Shape: {image.shape} (Z, Y, X)")
            print(f"  Data type: {image.dtype}")
            print(f"  Min value: {np.min(image)}")
            print(f"  Max value: {np.max(image)}")
            print(f"  Mean value: {np.mean(image):.2f}")
            print(f"  Std value: {np.std(image):.2f}")
            print(f"  Size: {image.size * image.dtype.itemsize / (1024**2):.2f} MB")
        
        # Load and inspect spots
        if spots_file.exists():
            print(f"\nSpot annotations:")
            spots = np.load(spots_file)
            print(f"  Number of spots: {len(spots)}")
            if len(spots) > 0:
                print(f"  Spot coordinates shape: {spots.shape}")
                print(f"  Coordinate ranges:")
                print(f"    Z: [{spots[:, 0].min()}, {spots[:, 0].max()}]")
                print(f"    Y: [{spots[:, 1].min()}, {spots[:, 1].max()}]")
                print(f"    X: [{spots[:, 2].min()}, {spots[:, 2].max()}]")
                
                # Check if spots are within image bounds
                if image_file.exists():
                    image = imread(image_file)
                    z_max, y_max, x_max = image.shape
                    valid_z = (spots[:, 0] >= 0) & (spots[:, 0] < z_max)
                    valid_y = (spots[:, 1] >= 0) & (spots[:, 1] < y_max)
                    valid_x = (spots[:, 2] >= 0) & (spots[:, 2] < x_max)
                    valid = valid_z & valid_y & valid_x
                    print(f"  Valid spots (within bounds): {valid.sum()} / {len(spots)}")
        
        # Check spot plot
        if spot_plot_file.exists():
            print(f"\nSpot plot:")
            spot_plot = imread(spot_plot_file)
            print(f"  Shape: {spot_plot.shape}")
            print(f"  Data type: {spot_plot.dtype}")
        else:
            print(f"\nSpot plot: Not found (may not have been cropped if dimensions didn't match)")
        
        print()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        regions_dir = Path(sys.argv[1])
    else:
        regions_dir = Path("/Users/eliasguan/Desktop/DL_210_data_analysis/ML_data_set_trainning/extracted_regions/regions")
    
    inspect_regions(regions_dir)

