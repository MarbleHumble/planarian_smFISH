#!/usr/bin/env python3
"""
test_thresholds_visualization_gpu.py
GPU-accelerated threshold testing with parallelization.

This version:
- Uses GPU spot detection (much faster than CPU Big-FISH)
- Parallelizes threshold testing
- Creates validation plots
"""

import numpy as np
from tifffile import imread
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

try:
    from functions.gpu_smfish_v2 import detect_spots_gpu_bigfish
    from functions.gpu_smfish_v2 import log_filter_gpu
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    print("Warning: GPU functions not available, will use CPU Big-FISH")

try:
    from bigfish.detection import detect_spots
    from bigfish.stack import log_filter
    HAS_BIGFISH = True
except ImportError:
    HAS_BIGFISH = False


def test_single_threshold_gpu(image, log_img, threshold, sigma, min_distance,
                              voxel_size, spot_radius, device="cuda"):
    """
    Test a single threshold using GPU detection.
    Returns detection results.
    """
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
        return spots
    except Exception as e:
        print(f"    Error with threshold {threshold:.2f}: {e}")
        return np.array([]).reshape(0, 3)


def test_single_threshold_cpu(image, threshold, sigma, min_distance,
                             voxel_size, spot_radius):
    """
    Test a single threshold using CPU Big-FISH.
    """
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
        return spots
    except Exception as e:
        print(f"    Error with threshold {threshold:.2f}: {e}")
        return np.array([]).reshape(0, 3)


def test_multiple_thresholds_parallel(image_path, annotation_path,
                                     thresholds=None, sigma=(1, 1.7, 1.7),
                                     min_distance=(3, 3, 3), voxel_size=(361, 75, 75),
                                     spot_radius=(600, 300, 300), use_gpu=True,
                                     device="cuda", n_workers=4):
    """
    Test multiple thresholds in parallel (GPU or CPU).
    """
    # Load image and annotations
    image = imread(image_path)
    gt_spots = np.load(annotation_path)
    
    # Generate LoG (once, reuse for all thresholds)
    if use_gpu and HAS_GPU:
        print(f"  Generating LoG on GPU...")
        log_img = log_filter_gpu(image, sigma, device=device)
    elif HAS_BIGFISH:
        print(f"  Generating LoG on CPU...")
        log_img = log_filter(image, kernel_size=sigma)
    else:
        raise ImportError("Need either GPU functions or Big-FISH")
    
    # Auto-generate thresholds if not provided
    if thresholds is None:
        log_flat = log_img.flatten()
        thresholds = [
            np.percentile(log_flat, p) for p in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
        ]
    
    print(f"  Testing {len(thresholds)} thresholds...")
    
    results = {}
    
    if use_gpu and HAS_GPU:
        # GPU version - test sequentially (GPU handles parallelism internally)
        for threshold in thresholds:
            detected_spots = test_single_threshold_gpu(
                image, log_img, threshold, sigma, min_distance,
                voxel_size, spot_radius, device=device
            )
            
            # Match with ground truth
            from scipy.spatial.distance import cdist
            if len(detected_spots) > 0 and len(gt_spots) > 0:
                distances = cdist(detected_spots, gt_spots, metric='euclidean')
                matches = 0
                matched_gt = set()
                for i in range(len(detected_spots)):
                    closest_gt = np.argmin(distances[i, :])
                    if distances[i, closest_gt] <= 3.0 and closest_gt not in matched_gt:
                        matches += 1
                        matched_gt.add(closest_gt)
                
                precision = matches / len(detected_spots) if len(detected_spots) > 0 else 0
                recall = matches / len(gt_spots) if len(gt_spots) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            else:
                precision = recall = f1 = 0.0
            
            results[threshold] = {
                'detected_spots': detected_spots,
                'n_detected': len(detected_spots),
                'n_gt': len(gt_spots),
                'precision': precision,
                'recall': recall,
                'f1': f1,
            }
    
    elif HAS_BIGFISH:
        # CPU version - can parallelize
        if n_workers > 1:
            print(f"  Using {n_workers} parallel workers...")
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(test_single_threshold_cpu, image, t, sigma,
                                  min_distance, voxel_size, spot_radius): t
                    for t in thresholds
                }
                
                for future in futures:
                    threshold = futures[future]
                    detected_spots = future.result()
                    
                    # Match with ground truth
                    from scipy.spatial.distance import cdist
                    if len(detected_spots) > 0 and len(gt_spots) > 0:
                        distances = cdist(detected_spots, gt_spots, metric='euclidean')
                        matches = 0
                        matched_gt = set()
                        for i in range(len(detected_spots)):
                            closest_gt = np.argmin(distances[i, :])
                            if distances[i, closest_gt] <= 3.0 and closest_gt not in matched_gt:
                                matches += 1
                                matched_gt.add(closest_gt)
                        
                        precision = matches / len(detected_spots) if len(detected_spots) > 0 else 0
                        recall = matches / len(gt_spots) if len(gt_spots) > 0 else 0
                        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    else:
                        precision = recall = f1 = 0.0
                    
                    results[threshold] = {
                        'detected_spots': detected_spots,
                        'n_detected': len(detected_spots),
                        'n_gt': len(gt_spots),
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                    }
        else:
            # Sequential CPU
            for threshold in thresholds:
                detected_spots = test_single_threshold_cpu(
                    image, threshold, sigma, min_distance, voxel_size, spot_radius
                )
                
                # Match with ground truth
                from scipy.spatial.distance import cdist
                if len(detected_spots) > 0 and len(gt_spots) > 0:
                    distances = cdist(detected_spots, gt_spots, metric='euclidean')
                    matches = 0
                    matched_gt = set()
                    for i in range(len(detected_spots)):
                        closest_gt = np.argmin(distances[i, :])
                        if distances[i, closest_gt] <= 3.0 and closest_gt not in matched_gt:
                            matches += 1
                            matched_gt.add(closest_gt)
                    
                    precision = matches / len(detected_spots) if len(detected_spots) > 0 else 0
                    recall = matches / len(gt_spots) if len(gt_spots) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                else:
                    precision = recall = f1 = 0.0
                
                results[threshold] = {
                    'detected_spots': detected_spots,
                    'n_detected': len(detected_spots),
                    'n_gt': len(gt_spots),
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                }
    
    return results, image, log_img, gt_spots


def create_validation_plot(image, log_img, gt_spots, results, 
                           output_path, max_thresholds_to_show=16):
    """
    Create comprehensive validation plot (same as CPU version).
    """
    # Select middle slice for visualization
    z_mid = image.shape[0] // 2
    
    # Sort thresholds by value
    thresholds = sorted(results.keys())
    n_thresh = min(len(thresholds), max_thresholds_to_show)
    thresholds_to_show = thresholds[:n_thresh]
    
    # Calculate grid size (4 columns)
    n_rows = (n_thresh // 4) + 1  # +1 for ground truth row
    if n_thresh % 4 == 0:
        n_rows += 1
    
    # Create figure
    fig = plt.figure(figsize=(20, 5 * n_rows))
    gs = GridSpec(n_rows, 4, figure=fig, hspace=0.4, wspace=0.3)
    
    # Plot ground truth on first row
    ax_gt = fig.add_subplot(gs[0, 0])
    ax_gt.imshow(image[z_mid], cmap='gray')
    if len(gt_spots) > 0:
        gt_in_slice = gt_spots[gt_spots[:, 0] == z_mid]
        if len(gt_in_slice) > 0:
            ax_gt.scatter(gt_in_slice[:, 2], gt_in_slice[:, 1], 
                         c='green', s=30, alpha=0.7, marker='o', 
                         edgecolors='darkgreen', linewidths=0.5,
                         label=f'GT ({len(gt_in_slice)})')
    ax_gt.set_title(f'Ground Truth\n{len(gt_spots)} spots total', fontsize=12, fontweight='bold')
    ax_gt.legend(fontsize=10)
    ax_gt.axis('off')
    
    # Plot each threshold result
    for idx, threshold in enumerate(thresholds_to_show):
        row = (idx // 4) + 1
        col = idx % 4
        
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(image[z_mid], cmap='gray')
        
        result = results[threshold]
        detected = result['detected_spots']
        
        if len(detected) > 0:
            det_in_slice = detected[detected[:, 0] == z_mid]
            if len(det_in_slice) > 0:
                ax.scatter(det_in_slice[:, 2], det_in_slice[:, 1],
                          c='red', s=30, alpha=0.7, marker='o',
                          edgecolors='darkred', linewidths=0.5,
                          label=f'Det ({len(det_in_slice)})')
        
        # Overlay ground truth (lighter)
        if len(gt_spots) > 0:
            gt_in_slice = gt_spots[gt_spots[:, 0] == z_mid]
            if len(gt_in_slice) > 0:
                ax.scatter(gt_in_slice[:, 2], gt_in_slice[:, 1],
                          c='green', s=15, alpha=0.4, marker='x',
                          linewidths=1, label='GT')
        
        # Title with metrics
        title = f'Thresh={threshold:.2f}\n'
        title += f'Det: {result["n_detected"]}, F1: {result["f1"]:.3f}\n'
        title += f'Prec: {result["precision"]:.3f}, Rec: {result["recall"]:.3f}'
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
        ax.axis('off')
    
    plt.suptitle(f'Threshold Validation (GPU) - {Path(output_path).stem}', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved visualization: {output_path.name}")


def batch_validate_thresholds(data_dir, output_dir, 
                              thresholds=None, n_thresholds=16,
                              use_gpu=True, device="cuda", n_workers=4):
    """
    Batch validate thresholds with GPU acceleration.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find images directory
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
    
    image_files = sorted(list(original_dir.glob('*.tif')))
    
    if len(image_files) == 0:
        raise ValueError(f"No .tif files found in {original_dir}")
    
    print(f"{'='*60}")
    print(f"Batch Threshold Validation ({'GPU' if use_gpu else 'CPU'})")
    print(f"{'='*60}")
    print(f"Found {len(image_files)} images")
    print(f"Output directory: {output_dir}")
    if use_gpu:
        print(f"Device: {device}")
    else:
        print(f"CPU workers: {n_workers}")
    print()
    
    all_results = {}
    
    for img_file in image_files:
        image_name = img_file.stem
        print(f"Processing: {image_name}")
        
        annotation_file = annotations_dir / f"{image_name}_spots.npy"
        if not annotation_file.exists():
            print(f"  Warning: No annotations found, skipping")
            continue
        
        try:
            # Test thresholds
            results, image, log_img, gt_spots = test_multiple_thresholds_parallel(
                img_file, annotation_file, thresholds=thresholds,
                use_gpu=use_gpu, device=device, n_workers=n_workers
            )
            
            print(f"    Tested {len(results)} thresholds")
            print(f"    Ground truth spots: {len(gt_spots)}")
            
            all_results[image_name] = {
                'results': results,
                'n_gt': len(gt_spots),
            }
            
            # Find best threshold
            best_f1 = -1
            best_threshold = None
            for threshold, result in results.items():
                if result['f1'] > best_f1:
                    best_f1 = result['f1']
                    best_threshold = threshold
            
            print(f"    Best threshold: {best_threshold:.2f} (F1: {best_f1:.3f})")
            
            # Create visualization
            output_plot = output_dir / f"{image_name}_threshold_validation.png"
            create_validation_plot(image, log_img, gt_spots, results, output_plot)
            print()
            
        except Exception as e:
            print(f"  Error processing {image_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create summary (same as CPU version)
    print(f"\n{'='*60}")
    print(f"Creating summary...")
    summary = {}
    for img_name, data in all_results.items():
        best_f1 = -1
        best_threshold = None
        best_metrics = {}
        
        for threshold, result in data['results'].items():
            if result['f1'] > best_f1:
                best_f1 = result['f1']
                best_threshold = threshold
                best_metrics = {
                    'precision': result['precision'],
                    'recall': result['recall'],
                    'f1': result['f1'],
                    'n_detected': result['n_detected'],
                }
        
        summary[img_name] = {
            'best_threshold': float(best_threshold) if best_threshold is not None else None,
            'best_f1': float(best_f1),
            'n_gt': data['n_gt'],
            **best_metrics
        }
    
    summary_file = output_dir / 'threshold_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_file}")
    
    # Print summary table
    print(f"\n{'='*60}")
    print(f"Summary of Best Thresholds")
    print(f"{'='*60}")
    print(f"{'Image':<40} {'Best Thresh':<15} {'F1':<8} {'Prec':<8} {'Rec':<8} {'Det':<8}")
    print(f"{'-'*90}")
    for img_name, data in sorted(summary.items()):
        print(f"{img_name:<40} {data['best_threshold']:>14.2f}  "
              f"{data['best_f1']:>7.3f}  {data['precision']:>7.3f}  "
              f"{data['recall']:>7.3f}  {data['n_detected']:>7d}")
    
    print(f"\n{'='*60}")
    print(f"Validation complete!")
    print(f"Plots saved to: {output_dir}")
    print(f"{'='*60}\n")
    
    return all_results, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GPU-accelerated threshold testing with visualization"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing Original Images/ and annotations/"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for plots and summary"
    )
    parser.add_argument(
        "--n_thresholds",
        type=int,
        default=16,
        help="Number of thresholds to test (default: 16)"
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs='+',
        default=None,
        help="Specific thresholds to test (overrides n_thresholds)"
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        default=True,
        help="Use GPU acceleration (default: True)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="GPU device (default: cuda)"
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=4,
        help="Number of CPU workers if not using GPU (default: 4)"
    )
    
    args = parser.parse_args()
    
    batch_validate_thresholds(
        args.data_dir, 
        args.output_dir, 
        thresholds=args.thresholds,
        n_thresholds=args.n_thresholds,
        use_gpu=args.use_gpu,
        device=args.device,
        n_workers=args.n_workers
    )

