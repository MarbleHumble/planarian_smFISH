#!/usr/bin/env python3
"""
prepare_region_training_data.py
Prepare training data from extracted regions for threshold prediction model.

This script:
1. Loads LoG filtered images (generates if needed)
2. Extracts features from regions
3. Finds optimal threshold for each region using ground truth spots
4. Prepares training dataset CSV
"""

import numpy as np
import pandas as pd
from tifffile import imread, imwrite
from pathlib import Path
import json
import argparse
from scipy.spatial.distance import cdist

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
try:
    from bigfish.stack import log_filter
    from bigfish.detection import detect_spots
    HAS_BIGFISH = True
except ImportError:
    HAS_BIGFISH = False
    log_filter = None
    print("Warning: bigfish not available, will use original image statistics instead of LoG")


def extract_spot_decay_features(log_img_region, ground_truth_spots, radius=5):
    """
    Extract features related to spot decay profile.
    
    smFISH spots have bright centers that decay outward.
    This function measures this decay characteristic.
    
    Args:
        log_img_region: LoG filtered image region
        ground_truth_spots: Validated spot coordinates (N, 3) in region coordinates
        radius: Radius to measure decay (pixels)
    
    Returns:
        Dictionary with decay-related features
    """
    if len(ground_truth_spots) == 0:
        return {
            'spot_decay_mean': 0.0,
            'spot_decay_std': 0.0,
            'spot_center_intensity_mean': 0.0,
            'spot_center_intensity_std': 0.0,
            'spot_surround_intensity_mean': 0.0,
            'spot_contrast_mean': 0.0,
            'spot_contrast_std': 0.0,
        }
    
    Z, Y, X = log_img_region.shape
    decays = []
    center_intensities = []
    surround_intensities = []
    contrasts = []
    
    for spot in ground_truth_spots:
        z, y, x = int(spot[0]), int(spot[1]), int(spot[2])
        
        # Check bounds
        if not (0 <= z < Z and 0 <= y < Y and 0 <= x < X):
            continue
        
        # Get center intensity
        center_int = log_img_region[z, y, x]
        center_intensities.append(center_int)
        
        # Get surrounding region
        y_min = max(0, y - radius)
        y_max = min(Y, y + radius + 1)
        x_min = max(0, x - radius)
        x_max = min(X, x + radius + 1)
        z_min = max(0, z - 1)
        z_max = min(Z, z + 2)
        
        # Extract 3D patch
        patch = log_img_region[z_min:z_max, y_min:y_max, x_min:x_max]
        
        # Distance from center
        zz, yy, xx = np.indices(patch.shape)
        zz = zz - (z - z_min)
        yy = yy - (y - y_min)
        xx = xx - (x - x_min)
        distances = np.sqrt(zz**2 + yy**2 + xx**2)
        
        # Remove center point
        mask = distances > 0.5
        if mask.sum() > 0:
            surround_int = np.mean(patch[mask])
            surround_intensities.append(surround_int)
            
            # Decay: difference between center and surround (should be positive for good spots)
            decay = center_int - surround_int
            decays.append(decay)
            
            # Contrast: (center - surround) / (surround + epsilon)
            contrast = (center_int - surround_int) / (abs(surround_int) + 1e-6)
            contrasts.append(contrast)
    
    # Compute statistics
    features = {
        'spot_decay_mean': float(np.mean(decays)) if len(decays) > 0 else 0.0,
        'spot_decay_std': float(np.std(decays)) if len(decays) > 0 else 0.0,
        'spot_center_intensity_mean': float(np.mean(center_intensities)) if len(center_intensities) > 0 else 0.0,
        'spot_center_intensity_std': float(np.std(center_intensities)) if len(center_intensities) > 0 else 0.0,
        'spot_surround_intensity_mean': float(np.mean(surround_intensities)) if len(surround_intensities) > 0 else 0.0,
        'spot_contrast_mean': float(np.mean(contrasts)) if len(contrasts) > 0 else 0.0,
        'spot_contrast_std': float(np.std(contrasts)) if len(contrasts) > 0 else 0.0,
    }
    
    return features


def extract_region_features(log_img_region, original_img_region=None, 
                           ground_truth_spots=None):
    """
    Extract comprehensive features from LoG image region.
    Now includes spot decay features if ground truth spots are provided.
    """
    features = {}
    
    # Basic statistics
    features['log_mean'] = float(np.mean(log_img_region))
    features['log_std'] = float(np.std(log_img_region))
    features['log_min'] = float(np.min(log_img_region))
    features['log_max'] = float(np.max(log_img_region))
    features['log_median'] = float(np.median(log_img_region))
    
    # Percentiles of LoG image
    for p in [1, 5, 10, 25, 50, 75, 85, 90, 95, 99]:
        features[f'log_p{p}'] = float(np.percentile(log_img_region, p))
    
    # Background characteristics
    bg_p85 = np.percentile(log_img_region, 85)
    bg_p95 = np.percentile(log_img_region, 95)
    features['background_p85'] = float(bg_p85)
    features['background_p95'] = float(bg_p95)
    features['background_level'] = float(np.mean(log_img_region[log_img_region > bg_p85]))
    
    # Region dimensions
    features['z_size'] = int(log_img_region.shape[0])
    features['y_size'] = int(log_img_region.shape[1])
    features['x_size'] = int(log_img_region.shape[2])
    features['total_voxels'] = int(log_img_region.size)
    
    # Histogram characteristics
    if HAS_SCIPY:
        features['hist_skew'] = float(stats.skew(log_img_region.flatten()))
        features['hist_kurtosis'] = float(stats.kurtosis(log_img_region.flatten()))
    else:
        # Fallback without scipy
        features['hist_skew'] = 0.0
        features['hist_kurtosis'] = 0.0
    
    # Dynamic range and contrast
    features['dynamic_range'] = float(features['log_max'] - features['log_min'])
    features['contrast'] = float(features['log_std'] / (abs(features['log_mean']) + 1e-10))
    
    # Original image stats if available
    if original_img_region is not None:
        features['original_mean'] = float(np.mean(original_img_region))
        features['original_std'] = float(np.std(original_img_region))
        features['original_max'] = float(np.max(original_img_region))
    else:
        features['original_mean'] = 0.0
        features['original_std'] = 0.0
        features['original_max'] = 0.0
    
    # Candidate spot statistics (using threshold=0 to get all minima)
    if HAS_BIGFISH:
        try:
            candidate_spots, _ = detect_spots(
                images=original_img_region if original_img_region is not None else log_img_region,
                threshold=0,  # Get all candidates
                return_threshold=True,
                voxel_size=(361, 75, 75),
                spot_radius=(600, 300, 300),
                log_kernel_size=(1, 1.7, 1.7),
                minimum_distance=(3, 3, 3),
            )
            
            if len(candidate_spots) > 0:
                log_values_at_candidates = log_img_region[
                    candidate_spots[:, 0],
                    candidate_spots[:, 1],
                    candidate_spots[:, 2]
                ]
                features['n_candidates'] = len(candidate_spots)
                features['candidate_log_mean'] = float(np.mean(log_values_at_candidates))
                features['candidate_log_std'] = float(np.std(log_values_at_candidates))
                features['candidate_log_min'] = float(np.min(log_values_at_candidates))
                features['candidate_log_median'] = float(np.median(log_values_at_candidates))
                for p in [5, 10, 25, 50, 75, 90, 95]:
                    features[f'candidate_log_p{p}'] = float(np.percentile(log_values_at_candidates, p))
            else:
                features['n_candidates'] = 0
                for key in ['candidate_log_mean', 'candidate_log_std', 'candidate_log_min',
                           'candidate_log_median'] + [f'candidate_log_p{p}' for p in [5, 10, 25, 50, 75, 90, 95]]:
                    features[key] = 0.0
        except Exception as e:
            features['n_candidates'] = 0
            for key in ['candidate_log_mean', 'candidate_log_std', 'candidate_log_min',
                       'candidate_log_median'] + [f'candidate_log_p{p}' for p in [5, 10, 25, 50, 75, 90, 95]]:
                features[key] = 0.0
    else:
        # Set default values if bigfish not available
        features['n_candidates'] = 0
        for key in ['candidate_log_mean', 'candidate_log_std', 'candidate_log_min',
                   'candidate_log_median'] + [f'candidate_log_p{p}' for p in [5, 10, 25, 50, 75, 90, 95]]:
            features[key] = 0.0
    
    # Add spot decay features if ground truth spots available
    if ground_truth_spots is not None and len(ground_truth_spots) > 0:
        decay_features = extract_spot_decay_features(
            log_img_region, ground_truth_spots, radius=5
        )
        features.update(decay_features)
    else:
        # Set default values
        for key in ['spot_decay_mean', 'spot_decay_std', 'spot_center_intensity_mean',
                   'spot_center_intensity_std', 'spot_surround_intensity_mean',
                   'spot_contrast_mean', 'spot_contrast_std']:
            features[key] = 0.0
    
    return features


def find_optimal_threshold_for_region(log_img_region, ground_truth_spots,
                                      original_img_region=None,
                                      threshold_range=None, tolerance=3.0,
                                      sigma=(1, 1.7, 1.7),
                                      min_distance=(3, 3, 3),
                                      voxel_size=(361, 75, 75),
                                      spot_radius=(600, 300, 300)):
    """
    Find optimal threshold that maximizes F1 score with ground truth spots.
    """
    if len(ground_truth_spots) == 0:
        return None, 0.0, {}
    
    # Try different thresholds
    if threshold_range is None:
        # Auto-determine range from LoG image statistics
        log_min = np.min(log_img_region)
        log_max = np.max(log_img_region)
        # Use a range that covers reasonable thresholds
        test_thresholds = np.linspace(log_min, log_max, 100)
    else:
        test_thresholds = np.linspace(threshold_range[0], threshold_range[1], 100)
    
    best_threshold = None
    best_f1 = -1
    best_metrics = {}
    
    # Use original image for detection if available, otherwise use LoG
    detection_image = original_img_region if original_img_region is not None else log_img_region
    
    for threshold in test_thresholds:
        try:
            # Detect spots with this threshold
            detected_spots, _ = detect_spots(
                images=detection_image,
                threshold=threshold,
                return_threshold=True,
                voxel_size=voxel_size,
                spot_radius=spot_radius,
                log_kernel_size=sigma,
                minimum_distance=min_distance,
            )
            
            if len(detected_spots) == 0:
                continue
            
            # Match detected spots with ground truth
            distances = cdist(detected_spots, ground_truth_spots, metric='euclidean')
            
            # Greedy matching
            matched_gt = set()
            matches = 0
            
            for i in range(len(detected_spots)):
                closest_gt = np.argmin(distances[i, :])
                if distances[i, closest_gt] <= tolerance and closest_gt not in matched_gt:
                    matches += 1
                    matched_gt.add(closest_gt)
            
            # Compute metrics
            precision = matches / len(detected_spots) if len(detected_spots) > 0 else 0
            recall = matches / len(ground_truth_spots) if len(ground_truth_spots) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_metrics = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'n_detected': len(detected_spots),
                    'n_gt': len(ground_truth_spots),
                    'n_matches': matches
                }
        except Exception as e:
            continue
    
    return best_threshold, best_f1, best_metrics


def prepare_training_data(regions_dir, output_csv, 
                         use_manual_threshold=True, manual_threshold=10.25,
                         find_optimal=True):
    """
    Prepare training data from extracted regions.
    
    Args:
        regions_dir: Directory containing region subdirectories
        output_csv: Output CSV file path
        use_manual_threshold: If True, use manual threshold from metadata
        manual_threshold: Manual threshold value if not in metadata
        find_optimal: If True, find optimal threshold using ground truth (takes longer)
    """
    regions_dir = Path(regions_dir)
    
    all_features = []
    all_thresholds = []
    all_metrics = []
    region_names = []
    
    region_dirs = sorted([d for d in regions_dir.iterdir() if d.is_dir()])
    
    print(f"Processing {len(region_dirs)} regions...")
    print()
    
    for region_dir in region_dirs:
        region_name = region_dir.name
        print(f"Processing: {region_name}")
        
        # Check for required files
        image_file = region_dir / 'image_region.tif'
        spots_file = region_dir / 'spots.npy'
        metadata_file = region_dir / 'region_info.json'
        
        if not image_file.exists():
            print(f"  Warning: Image file not found, skipping")
            continue
        
        if not spots_file.exists():
            print(f"  Warning: Spots file not found, skipping")
            continue
        
        # Load data
        print(f"  Loading image region...")
        image_region = imread(image_file)
        print(f"    Shape: {image_region.shape}")
        
        print(f"  Loading ground truth spots...")
        gt_spots = np.load(spots_file)
        print(f"    {len(gt_spots)} ground truth spots")
        
        if len(gt_spots) == 0:
            print(f"  Warning: No ground truth spots, skipping")
            continue
        
        # For training features, we'll use the original image statistics
        # (In production, LoG features would be better, but we can train with what we have)
        print(f"  Using original image for feature extraction...")
        log_img_region = image_region.astype(np.float32)
        
        # Extract features (including spot decay from ground truth)
        print(f"  Extracting features...")
        features = extract_region_features(log_img_region, image_region, gt_spots)
        
        # Get threshold
        if use_manual_threshold:
            # Try to get from metadata
            threshold = manual_threshold
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                    threshold = metadata.get('threshold', manual_threshold)
            
            print(f"  Using manual threshold: {threshold}")
            optimal_thresh = threshold
            f1 = 0.0
            metrics = {}
        else:
            threshold = manual_threshold  # Fallback
        
        if find_optimal and HAS_BIGFISH:
            print(f"  Finding optimal threshold (this may take a while)...")
            optimal_thresh, f1, metrics = find_optimal_threshold_for_region(
                log_img_region, gt_spots, image_region,
                tolerance=3.0
            )
            
            if optimal_thresh is None:
                print(f"  Warning: Could not find optimal threshold, using manual: {threshold}")
                optimal_thresh = threshold
                f1 = 0.0
                metrics = {}
            else:
                print(f"  Optimal threshold: {optimal_thresh:.4f}, F1: {f1:.4f}")
        elif find_optimal and not HAS_BIGFISH:
            print(f"  Warning: bigfish not available, cannot find optimal threshold, using manual: {threshold}")
            optimal_thresh = threshold
            f1 = 0.0
            metrics = {}
        
        # Store data
        all_features.append(features)
        all_thresholds.append(optimal_thresh)
        all_metrics.append(metrics)
        region_names.append(region_name)
        
        print()
    
    # Create DataFrame
    print(f"\nCreating training dataset...")
    df = pd.DataFrame(all_features)
    df['threshold'] = all_thresholds
    df['region_name'] = region_names
    
    # Add metrics if available
    if find_optimal and len(all_metrics) > 0 and len(all_metrics[0]) > 0:
        for key in ['precision', 'recall', 'f1', 'n_detected', 'n_gt', 'n_matches']:
            df[f'metric_{key}'] = [m.get(key, 0) for m in all_metrics]
    
    # Save
    output_csv = Path(output_csv)
    df.to_csv(output_csv, index=False)
    
    print(f"\n{'='*60}")
    print(f"Training data prepared!")
    print(f"{'='*60}")
    print(f"Total regions: {len(df)}")
    print(f"Features: {len(df.columns) - 2 - (6 if find_optimal else 0)}")  # Excluding threshold, region_name, and metrics
    print(f"Threshold range: [{df['threshold'].min():.4f}, {df['threshold'].max():.4f}]")
    if find_optimal and 'metric_f1' in df.columns:
        print(f"Average F1: {df['metric_f1'].mean():.4f}")
    print(f"Saved to: {output_csv}")
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare training data from extracted regions"
    )
    parser.add_argument(
        "--regions_dir",
        type=str,
        required=True,
        help="Directory containing region subdirectories"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="region_training_data.csv",
        help="Output CSV filename"
    )
    parser.add_argument(
        "--use_manual_threshold",
        action="store_true",
        default=True,
        help="Use manual threshold from metadata (default: True)"
    )
    parser.add_argument(
        "--find_optimal",
        action="store_true",
        help="Also find optimal threshold using ground truth (takes longer)"
    )
    parser.add_argument(
        "--manual_threshold",
        type=float,
        default=10.25,
        help="Manual threshold value if not in metadata"
    )
    
    args = parser.parse_args()
    
    df = prepare_training_data(
        args.regions_dir,
        args.output,
        use_manual_threshold=args.use_manual_threshold,
        manual_threshold=args.manual_threshold,
        find_optimal=args.find_optimal
    )
    
    print("\nTraining data preview:")
    print(df[['threshold', 'region_name']].head())

