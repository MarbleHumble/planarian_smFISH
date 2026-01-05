#!/usr/bin/env python3
"""
Run Big-FISH spot detection with threshold=10.25 and create spot plot.
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


def generate_coordinates_2D(y, x, shape, iteration=4, get_inner_spot=False):
    """
    Generate coordinates for 2D spot plotting.
    Creates a circular pattern around (y, x) coordinate.
    
    Args:
        y, x: Center coordinates
        shape: (height, width) of the image
        iteration: Number of iterations (determines radius)
        get_inner_spot: If True, return all coordinates; if False, return only outer edge
    
    Returns:
        List of (y, x) coordinate tuples
    """
    coordinates_collection = [(y, x)]
    max_y = shape[0] - 1
    max_x = shape[1] - 1

    for _ in range(iteration):
        current_coordinates = coordinates_collection.copy()
        
        for coord in current_coordinates:
            cy, cx = coord
            
            # Add neighboring coordinates if within bounds
            if cx + 1 <= max_x:
                coordinates_collection.append((cy, cx + 1))
            if 0 <= cx - 1:
                coordinates_collection.append((cy, cx - 1))
            if cy + 1 <= max_y:
                coordinates_collection.append((cy + 1, cx))
            if 0 <= cy - 1:
                coordinates_collection.append((cy - 1, cx))
    
    # Remove duplicates
    coordinates_collection = list(set(coordinates_collection))
    
    if get_inner_spot:
        return coordinates_collection
    else:
        # Return only outer edge (Manhattan distance == iteration)
        coordinates_collection = [
            coord for coord in coordinates_collection 
            if abs(coord[0] - y) + abs(coord[1] - x) == iteration
        ]
        return coordinates_collection


def create_spot_plot(image, spots, plot_spot_size=4, plot_inner_circle=False, plot_spot_label=False):
    """
    Create a 3D spot plot from detected spots.
    
    Args:
        image: 3D image array (Z, Y, X)
        spots: Nx3 array of spot coordinates (z, y, x)
        plot_spot_size: Size of spot plot (number of iterations)
        plot_inner_circle: If True, fill inner circle; if False, only outer edge
        plot_spot_label: If True, use uint32 and label each spot; if False, use uint8 and set to 255
    
    Returns:
        3D array with spot plot
    """
    z, y, x = image.shape
    
    if plot_spot_label:
        spot_plot = np.zeros(image.shape, dtype=np.uint32)
    else:
        spot_plot = np.zeros(image.shape, dtype=np.uint8)
    
    shape_2d = [spot_plot.shape[1], spot_plot.shape[2]]  # [Y, X]
    
    for i, spot in enumerate(spots):
        z_coord = int(spot[0])
        y_coord = int(spot[1])
        x_coord = int(spot[2])
        
        # Get plot coordinates in 2D
        plot_locations = generate_coordinates_2D(
            y_coord, x_coord, 
            shape_2d, 
            iteration=plot_spot_size,
            get_inner_spot=plot_inner_circle
        )
        
        # Draw spots
        for plot_y, plot_x in plot_locations:
            if 0 <= z_coord < z and 0 <= plot_y < y and 0 <= plot_x < x:
                if plot_spot_label:
                    spot_plot[z_coord, plot_y, plot_x] = i + 1
                else:
                    spot_plot[z_coord, plot_y, plot_x] = 255
    
    return spot_plot


def main():
    # Image path
    image_path = "/Volumes/Backup Plus/DL_210_data_analysis/Test image/12hr_Amputation_Image2_565.tif"
    config_path = "config.yaml"
    threshold = 10.25
    
    # Load config
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)
    
    # Get parameters from config
    sigma = tuple(config.get('kernel_size', [1, 1.5, 1.5]))
    min_distance = tuple(config.get('minimal_distance', [2, 2, 2]))
    voxel_size = tuple(config.get('voxel_size', [361, 75, 75]))
    spot_radius = tuple(config.get('spot_size', [600, 300, 300]))
    plot_spot_size = config.get('plot_spot_size', 4)
    plot_inner_circle = config.get('plotInnerCircle', False)
    plot_spot_label = config.get('plotSpotLabel', False)
    
    print("="*60)
    print("Big-FISH Spot Detection")
    print("="*60)
    print(f"Image: {image_path}")
    print(f"Threshold: {threshold}")
    print(f"Parameters:")
    print(f"  sigma (kernel_size): {sigma}")
    print(f"  min_distance: {min_distance}")
    print(f"  voxel_size: {voxel_size}")
    print(f"  spot_radius: {spot_radius}")
    print(f"  plot_spot_size: {plot_spot_size}")
    print("="*60)
    
    # Load image
    print("\nLoading image...")
    image = imread(image_path)
    print(f"Image shape: {image.shape} [Z, Y, X]")
    
    # Detect spots
    print(f"\nDetecting spots with threshold={threshold}...")
    detected_spots, threshold_used = detect_spots(
        images=image,
        threshold=threshold,
        return_threshold=True,
        voxel_size=voxel_size,
        spot_radius=spot_radius,
        log_kernel_size=sigma,
        minimum_distance=min_distance,
    )
    
    print(f"Detected {len(detected_spots)} spots")
    print(f"Threshold used: {threshold_used:.4f}")
    
    # Create spot plot
    print(f"\nCreating spot plot...")
    spot_plot = create_spot_plot(
        image, 
        detected_spots,
        plot_spot_size=plot_spot_size,
        plot_inner_circle=plot_inner_circle,
        plot_spot_label=plot_spot_label
    )
    
    # Save spot plot in same folder as image
    image_dir = Path(image_path).parent
    output_path = image_dir / "spotPlot.tif"
    
    print(f"\nSaving spot plot to: {output_path}")
    imwrite(output_path, spot_plot, photometric='minisblack')
    
    print(f"\nDone! Spot plot saved to: {output_path}")
    print("="*60)


if __name__ == "__main__":
    main()

