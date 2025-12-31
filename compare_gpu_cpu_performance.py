#!/usr/bin/env python3
"""
Quick performance test to compare GPU vs CPU for threshold testing.
Run this first to decide where to run the full validation.
"""

import numpy as np
import time
from pathlib import Path
from tifffile import imread
import torch

try:
    from functions.gpu_smfish_v2 import detect_spots_gpu_bigfish, log_filter_gpu
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

try:
    from bigfish.detection import detect_spots
    from bigfish.stack import log_filter
    HAS_BIGFISH = True
except ImportError:
    HAS_BIGFISH = False


def test_gpu_performance(image_path, device="cuda", n_tests=5):
    """Test GPU performance"""
    if not HAS_GPU:
        return None
    
    print(f"\n{'='*60}")
    print(f"Testing GPU Performance ({device})")
    print(f"{'='*60}")
    
    # Check device availability
    if device == "cuda":
        if not torch.cuda.is_available():
            print("CUDA not available!")
            return None
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    elif device == "mps":
        if not torch.backends.mps.is_available():
            print("MPS not available!")
            return None
        print("MPS (Metal) available")
    
    image = imread(image_path)
    print(f"Image shape: {image.shape}")
    print(f"Image size: {image.nbytes / 1024**2:.2f} MB")
    
    sigma = (1, 1.7, 1.7)
    min_distance = (3, 3, 3)
    voxel_size = (361, 75, 75)
    spot_radius = (600, 300, 300)
    
    # Generate LoG once
    print("\nGenerating LoG filter...")
    start = time.time()
    log_img = log_filter_gpu(image, sigma, device=device)
    log_time = time.time() - start
    print(f"LoG generation: {log_time:.2f} seconds")
    
    # Test detection speed
    thresholds = np.linspace(-50, -20, n_tests)
    times = []
    
    print(f"\nTesting {n_tests} detections...")
    for i, threshold in enumerate(thresholds):
        start = time.time()
        try:
            spots, _ = detect_spots_gpu_bigfish(
                image_np=image,
                sigma=sigma,
                min_distance=min_distance,
                threshold=threshold,
                voxel_size=voxel_size,
                spot_radius=spot_radius,
                device=device,
                return_threshold=True,
            )
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"  Threshold {threshold:.2f}: {elapsed:.2f}s ({len(spots)} spots)")
        except Exception as e:
            print(f"  Threshold {threshold:.2f}: Error - {e}")
            return None
    
    avg_time = np.mean(times)
    total_time = np.sum(times)
    
    print(f"\n{'='*60}")
    print(f"GPU Performance Summary")
    print(f"{'='*60}")
    print(f"Average detection time: {avg_time:.2f} seconds")
    print(f"Total time for {n_tests} detections: {total_time:.2f} seconds")
    print(f"Estimated time for 16 thresholds: {avg_time * 16:.2f} seconds")
    print(f"Estimated time for 18 images × 16 thresholds: {avg_time * 16 * 18 / 60:.1f} minutes")
    
    return {
        'device': device,
        'log_time': log_time,
        'avg_detection_time': avg_time,
        'total_time': total_time,
        'estimated_full_time_minutes': avg_time * 16 * 18 / 60
    }


def test_cpu_performance(image_path, n_tests=5):
    """Test CPU Big-FISH performance"""
    if not HAS_BIGFISH:
        return None
    
    print(f"\n{'='*60}")
    print(f"Testing CPU Big-FISH Performance")
    print(f"{'='*60}")
    
    image = imread(image_path)
    print(f"Image shape: {image.shape}")
    print(f"Image size: {image.nbytes / 1024**2:.2f} MB")
    
    sigma = (1, 1.7, 1.7)
    min_distance = (3, 3, 3)
    voxel_size = (361, 75, 75)
    spot_radius = (600, 300, 300)
    
    # Generate LoG once
    print("\nGenerating LoG filter...")
    start = time.time()
    log_img = log_filter(image, kernel_size=sigma)
    log_time = time.time() - start
    print(f"LoG generation: {log_time:.2f} seconds")
    
    # Test detection speed
    thresholds = np.linspace(-50, -20, n_tests)
    times = []
    
    print(f"\nTesting {n_tests} detections...")
    for i, threshold in enumerate(thresholds):
        start = time.time()
        try:
            spots, _ = detect_spots(
                images=image,
                threshold=threshold,
                return_threshold=True,
                voxel_size=voxel_size,
                spot_radius=spot_radius,
                log_kernel_size=sigma,
                minimum_distance=min_distance,
            )
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"  Threshold {threshold:.2f}: {elapsed:.2f}s ({len(spots)} spots)")
        except Exception as e:
            print(f"  Threshold {threshold:.2f}: Error - {e}")
            return None
    
    avg_time = np.mean(times)
    total_time = np.sum(times)
    
    print(f"\n{'='*60}")
    print(f"CPU Performance Summary")
    print(f"{'='*60}")
    print(f"Average detection time: {avg_time:.2f} seconds")
    print(f"Total time for {n_tests} detections: {total_time:.2f} seconds")
    print(f"Estimated time for 16 thresholds: {avg_time * 16:.2f} seconds")
    print(f"Estimated time for 18 images × 16 thresholds: {avg_time * 16 * 18 / 60:.1f} minutes")
    
    return {
        'device': 'cpu',
        'log_time': log_time,
        'avg_detection_time': avg_time,
        'total_time': total_time,
        'estimated_full_time_minutes': avg_time * 16 * 18 / 60
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare GPU vs CPU performance")
    parser.add_argument("--image_path", type=str, required=True,
                       help="Path to a test image")
    parser.add_argument("--n_tests", type=int, default=5,
                       help="Number of threshold tests to run")
    
    args = parser.parse_args()
    
    results = {}
    
    # Test MPS (Mac GPU)
    if HAS_GPU:
        try:
            results['mps'] = test_gpu_performance(args.image_path, device="mps", n_tests=args.n_tests)
        except Exception as e:
            print(f"MPS test failed: {e}")
            results['mps'] = None
    
    # Test CUDA (if available)
    if HAS_GPU:
        try:
            results['cuda'] = test_gpu_performance(args.image_path, device="cuda", n_tests=args.n_tests)
        except Exception as e:
            print(f"CUDA test failed: {e}")
            results['cuda'] = None
    
    # Test CPU
    if HAS_BIGFISH:
        try:
            results['cpu'] = test_cpu_performance(args.image_path, n_tests=args.n_tests)
        except Exception as e:
            print(f"CPU test failed: {e}")
            results['cpu'] = None
    
    # Print comparison
    print(f"\n{'='*60}")
    print(f"Performance Comparison")
    print(f"{'='*60}")
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if len(valid_results) == 0:
        print("No valid results!")
    else:
        print(f"{'Device':<10} {'Avg Time (s)':<15} {'Est. Full (min)':<20}")
        print(f"{'-'*45}")
        for device, result in sorted(valid_results.items(), 
                                    key=lambda x: x[1]['estimated_full_time_minutes']):
            print(f"{device:<10} {result['avg_detection_time']:>13.2f}  "
                  f"{result['estimated_full_time_minutes']:>18.1f}")
        
        # Recommendation
        best = min(valid_results.items(), 
                  key=lambda x: x[1]['estimated_full_time_minutes'])
        print(f"\n{'='*60}")
        print(f"RECOMMENDATION: Use {best[0].upper()}")
        print(f"Estimated time: {best[1]['estimated_full_time_minutes']:.1f} minutes")
        print(f"{'='*60}")

