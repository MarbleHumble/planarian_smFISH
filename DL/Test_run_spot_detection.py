"""
The script compares the original and optimized spot detection implementations. It:
1. Loads the image once and prepares it for both versions.
2. Runs both versions with the same parameters (original using Big-FISH, optimized using the CPU/GPU version).
3. Times each run and counts detected spots.
4. Prints a comparison table showing time, spot counts, speedup, and differences.
5. Saves both spot coordinate arrays to separate .npy files.
"""

import numpy as np
import sys
import os
import time
from PIL import Image

# Disable PIL decompression bomb warning for large images
Image.MAX_IMAGE_PIXELS = None

# Add your project path
sys.path.insert(0, '/Users/dennisliu/Documents/GitHub/planarian_smFISH')

# Import both versions
# Original module exposes 'detect_spots_from_config', not a direct 'detect_spots' API
from DL.original_spot_detection import detect_spots_from_config as detect_spots_original_config
from DL.DL_optimized_1_spot_detection import detect_spots as detect_spots_optimized

def load_image(image_path):
    """Load image from PNG or TIF file."""
    ext = os.path.splitext(image_path)[1].lower()
    
    if ext == '.png':
        from PIL import Image
        image = np.array(Image.open(image_path))
    elif ext in ['.tif', '.tiff']:
        import tifffile
        image = tifffile.imread(image_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    # Convert to appropriate dtype
    if image.dtype not in [np.uint8, np.uint16, np.float32, np.float64]:
        image = image.astype(np.float32)
    
    return image

def detect_spots_with_timing(detect_spots_func, image, version_name, log_kernel_size=1.5, minimum_distance=2):
    """Run spot detection with timing."""
    print(f"\n{'='*60}")
    print(f"Running {version_name}...")
    print(f"{'='*60}")
    
    detection_start = time.time()
    spots = detect_spots_func(
        image,
        log_kernel_size=log_kernel_size,
        minimum_distance=minimum_distance,
        threshold=None,              # Auto threshold
        remove_duplicate=True
    )
    detection_time = time.time() - detection_start
    
    num_spots = len(spots)
    print(f"✓ Detected {num_spots} spots")
    print(f"⏱️  Detection time: {detection_time:.2f} seconds")
    
    return spots, detection_time, num_spots

def run_original_wrapper(image, image_path, log_kernel_size=1.5, minimum_distance=2):
    """
    Adapter to call the original module's detect_spots_from_config using the same parameters.
    We construct a minimal config and force CPU (Big-FISH) backend for fair comparison.
    """
    ndim = image.ndim
    # Build config expected by original module
    config = {
        # Kernel/min-distance used by Big-FISH; provide as tuples per dimension
        "kernel_size": (log_kernel_size,) * ndim,
        "minimal_distance": (minimum_distance,) * ndim,
        # These are required by the original wrapper but won't override kernel/min-distance usage
        "voxel_size": (1,) * ndim,
        "spot_size": (1,) * ndim,
        # Ensure CPU path for apples-to-apples comparison
        "use_gpu": False,
        # No manual experiment threshold (use auto threshold)
        "experimentThreshold": None,
    }
    results_folder = os.path.join(os.path.dirname(image_path), "results_original")
    os.makedirs(results_folder, exist_ok=True)
    # Original wrapper expects a TIFF path; convert non-TIFF inputs to a temporary TIFF
    ext = os.path.splitext(image_path)[1].lower()
    path_for_original = image_path
    if ext not in [".tif", ".tiff"]:
        from tifffile import imwrite
        temp_tif = os.path.join(results_folder, "converted_input_for_original.tif")
        imwrite(temp_tif, image)
        path_for_original = temp_tif
    spots_exp, *_ = detect_spots_original_config(config, img_path=path_for_original, results_folder=results_folder)
    return spots_exp
# ##########################################################################################
# Test script for spot detection
# ##########################################################################################
if __name__ == "__main__":
    # Your image path
    image_path = "/Users/dennisliu/Documents/Image Processing/MAX_merged_BMP_magenta_Nucleus_blue.png"
    
    # Parameters
    log_kernel_size = 1.5    # Adjust based on your spot size
    minimum_distance = 2     # Adjust based on expected spot spacing
    
    # Load image once (shared between both versions)
    print(f"\n{'='*60}")
    print("Loading image...")
    print(f"{'='*60}")
    load_start = time.time()
    image = load_image(image_path)
    load_time = time.time() - load_start
    print(f"Image shape: {image.shape}, dtype: {image.dtype}")
    print(f"Image loading time: {load_time:.2f} seconds")
    
    # Run ORIGINAL version
    spots_original, time_original, num_spots_original = detect_spots_with_timing(
        # Lambda adapts the unified call signature to the original config-based API
        lambda img, log_kernel_size, minimum_distance, **_: run_original_wrapper(
            img, image_path, log_kernel_size, minimum_distance
        ),
        image,
        "ORIGINAL Spot Detection",
        log_kernel_size,
        minimum_distance
    )
    
    # Run OPTIMIZED version
    spots_optimized, time_optimized, num_spots_optimized = detect_spots_with_timing(
        detect_spots_optimized,
        image,
        "OPTIMIZED Spot Detection (CPU/GPU)",
        log_kernel_size,
        minimum_distance
    )
    
    # Comparison Summary
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Metric':<30} {'Original':<20} {'Optimized':<20} {'Difference':<20}")
    print(f"{'-'*90}")
    
    # Time comparison
    time_diff = time_original - time_optimized
    time_speedup = time_original / time_optimized if time_optimized > 0 else 0
    print(f"{'Detection Time (s)':<30} {time_original:<20.2f} {time_optimized:<20.2f} {time_diff:+.2f} ({time_speedup:.2f}x faster)")
    
    # Spots comparison
    spots_diff = num_spots_optimized - num_spots_original
    spots_pct_diff = (spots_diff / num_spots_original * 100) if num_spots_original > 0 else 0
    print(f"{'Number of Spots':<30} {num_spots_original:<20} {num_spots_optimized:<20} {spots_diff:+d} ({spots_pct_diff:+.2f}%)")
    
    # Accuracy check
    if num_spots_original == num_spots_optimized:
        print(f"\n✓ Both versions detected the SAME number of spots (accuracy maintained)")
    else:
        print(f"\n⚠️  Different number of spots detected:")
        print(f"   Original: {num_spots_original} spots")
        print(f"   Optimized: {num_spots_optimized} spots")
        print(f"   Difference: {abs(spots_diff)} spots ({abs(spots_pct_diff):.2f}%)")
    
    # Performance summary
    print(f"\n{'='*60}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    if time_speedup > 1:
        print(f"🚀 Optimized version is {time_speedup:.2f}x FASTER")
        print(f"   Time saved: {time_diff:.2f} seconds ({time_diff/time_original*100:.1f}% reduction)")
    elif time_speedup < 1:
        print(f"⚠️  Optimized version is {1/time_speedup:.2f}x SLOWER")
        print(f"   Time increase: {abs(time_diff):.2f} seconds")
    else:
        print(f"⚖️  Both versions have similar performance")
    
    print(f"{'='*60}\n")
    
    # Save results
    print("Saving results...")
    np.save("detected_spots_original.npy", spots_original)
    np.save("detected_spots_optimized.npy", spots_optimized)
    print("✓ Saved original spots to: detected_spots_original.npy")
    print("✓ Saved optimized spots to: detected_spots_optimized.npy")