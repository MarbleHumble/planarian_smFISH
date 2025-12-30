#!/usr/bin/env python3
"""
Compare GPU Big-FISH results with CPU Big-FISH results.

This script loads two .npy files containing spot coordinates and compares them:
1. Counts of spots
2. Distance-based matching (spots within tolerance)
3. Statistical comparison

Usage:
    python compare_results.py \
        --gpu_results results/experiment_spots.npy \
        --cpu_results /path/to/cpu/bigfish/spots_exp.npy \
        --tolerance 3.0
"""

import argparse
import numpy as np
from scipy.spatial.distance import cdist
import sys


def load_spots(filepath):
    """Load spot coordinates from .npy file."""
    try:
        spots = np.load(filepath)
        print(f"Loaded {len(spots)} spots from {filepath}")
        return spots
    except Exception as e:
        print(f"ERROR: Failed to load {filepath}: {e}", file=sys.stderr)
        sys.exit(1)


def match_spots(spots1, spots2, tolerance=3.0):
    """
    Match spots between two sets using distance-based matching.
    
    Args:
        spots1: First set of spots (N1, 3)
        spots2: Second set of spots (N2, 3)
        tolerance: Maximum distance for matching (in pixels)
        
    Returns:
        matched_pairs: List of (idx1, idx2) tuples for matched spots
        unmatched1: Indices of spots in spots1 that weren't matched
        unmatched2: Indices of spots in spots2 that weren't matched
    """
    if len(spots1) == 0 or len(spots2) == 0:
        return [], list(range(len(spots1))), list(range(len(spots2)))
    
    # Compute pairwise distances
    distances = cdist(spots1, spots2, metric='euclidean')
    
    # Find matches: each spot in spots1 matches to closest spot in spots2 if within tolerance
    matched_pairs = []
    matched2 = set()
    
    for i in range(len(spots1)):
        closest_idx = np.argmin(distances[i, :])
        if distances[i, closest_idx] <= tolerance and closest_idx not in matched2:
            matched_pairs.append((i, closest_idx))
            matched2.add(closest_idx)
    
    # Find unmatched spots
    matched1 = set(pair[0] for pair in matched_pairs)
    unmatched1 = [i for i in range(len(spots1)) if i not in matched1]
    unmatched2 = [i for i in range(len(spots2)) if i not in matched2]
    
    return matched_pairs, unmatched1, unmatched2


def compute_statistics(spots1, spots2, matched_pairs, unmatched1, unmatched2):
    """Compute comparison statistics."""
    stats = {}
    
    stats['count1'] = len(spots1)
    stats['count2'] = len(spots2)
    stats['count_diff'] = len(spots1) - len(spots2)
    stats['count_diff_percent'] = (stats['count_diff'] / len(spots2) * 100) if len(spots2) > 0 else 0
    
    stats['matched_count'] = len(matched_pairs)
    stats['unmatched1_count'] = len(unmatched1)
    stats['unmatched2_count'] = len(unmatched2)
    
    if len(matched_pairs) > 0:
        stats['match_rate1'] = len(matched_pairs) / len(spots1) * 100
        stats['match_rate2'] = len(matched_pairs) / len(spots2) * 100
    else:
        stats['match_rate1'] = 0
        stats['match_rate2'] = 0
    
    return stats


def compute_match_distances(spots1, spots2, matched_pairs):
    """Compute distances for matched pairs."""
    if len(matched_pairs) == 0:
        return np.array([])
    
    distances = []
    for idx1, idx2 in matched_pairs:
        dist = np.linalg.norm(spots1[idx1] - spots2[idx2])
        distances.append(dist)
    
    return np.array(distances)


def main():
    parser = argparse.ArgumentParser(
        description="Compare GPU and CPU Big-FISH spot detection results"
    )
    parser.add_argument(
        "--gpu_results",
        type=str,
        required=True,
        help="Path to GPU results .npy file"
    )
    parser.add_argument(
        "--cpu_results",
        type=str,
        required=True,
        help="Path to CPU Big-FISH results .npy file"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=3.0,
        help="Maximum distance for matching spots (pixels, default: 3.0)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional: Save comparison report to file"
    )
    
    args = parser.parse_args()
    
    # Load results
    print("=" * 60)
    print("Comparing GPU and CPU Big-FISH Results")
    print("=" * 60)
    print()
    
    spots_gpu = load_spots(args.gpu_results)
    spots_cpu = load_spots(args.cpu_results)
    print()
    
    # Match spots
    print(f"Matching spots with tolerance = {args.tolerance} pixels...")
    matched_pairs, unmatched_gpu, unmatched_cpu = match_spots(
        spots_gpu, spots_cpu, tolerance=args.tolerance
    )
    print(f"Found {len(matched_pairs)} matched spot pairs")
    print()
    
    # Compute statistics
    stats = compute_statistics(spots_gpu, spots_cpu, matched_pairs, unmatched_gpu, unmatched_cpu)
    
    # Compute match distances
    match_distances = compute_match_distances(spots_gpu, spots_cpu, matched_pairs)
    
    # Print report
    print("=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print()
    print(f"GPU Results:      {stats['count1']:>8} spots")
    print(f"CPU Results:      {stats['count2']:>8} spots")
    print(f"Difference:       {stats['count_diff']:>8} spots ({stats['count_diff_percent']:+.2f}%)")
    print()
    print(f"Matched spots:    {stats['matched_count']:>8} pairs")
    print(f"  - Match rate (GPU): {stats['match_rate1']:>6.2f}%")
    print(f"  - Match rate (CPU): {stats['match_rate2']:>6.2f}%")
    print()
    print(f"Unmatched GPU:    {stats['unmatched1_count']:>8} spots")
    print(f"Unmatched CPU:    {stats['unmatched2_count']:>8} spots")
    print()
    
    if len(match_distances) > 0:
        print("Match Distance Statistics (pixels):")
        print(f"  Mean:   {np.mean(match_distances):.3f}")
        print(f"  Median: {np.median(match_distances):.3f}")
        print(f"  Std:    {np.std(match_distances):.3f}")
        print(f"  Min:    {np.min(match_distances):.3f}")
        print(f"  Max:    {np.max(match_distances):.3f}")
        print()
    
    # Additional analysis: coordinate statistics
    print("Coordinate Statistics:")
    print(f"GPU spots - Z: [{spots_gpu[:, 0].min():.1f}, {spots_gpu[:, 0].max():.1f}], "
          f"Y: [{spots_gpu[:, 1].min():.1f}, {spots_gpu[:, 1].max():.1f}], "
          f"X: [{spots_gpu[:, 2].min():.1f}, {spots_gpu[:, 2].max():.1f}]")
    print(f"CPU spots - Z: [{spots_cpu[:, 0].min():.1f}, {spots_cpu[:, 0].max():.1f}], "
          f"Y: [{spots_cpu[:, 1].min():.1f}, {spots_cpu[:, 1].max():.1f}], "
          f"X: [{spots_cpu[:, 2].min():.1f}, {spots_cpu[:, 2].max():.1f}]")
    print()
    
    # Save report if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write("GPU vs CPU Big-FISH Comparison Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"GPU Results:      {stats['count1']:>8} spots\n")
            f.write(f"CPU Results:      {stats['count2']:>8} spots\n")
            f.write(f"Difference:       {stats['count_diff']:>8} spots ({stats['count_diff_percent']:+.2f}%)\n\n")
            f.write(f"Matched spots:    {stats['matched_count']:>8} pairs\n")
            f.write(f"  - Match rate (GPU): {stats['match_rate1']:>6.2f}%\n")
            f.write(f"  - Match rate (CPU): {stats['match_rate2']:>6.2f}%\n\n")
            f.write(f"Unmatched GPU:    {stats['unmatched1_count']:>8} spots\n")
            f.write(f"Unmatched CPU:    {stats['unmatched2_count']:>8} spots\n\n")
            if len(match_distances) > 0:
                f.write("Match Distance Statistics (pixels):\n")
                f.write(f"  Mean:   {np.mean(match_distances):.3f}\n")
                f.write(f"  Median: {np.median(match_distances):.3f}\n")
                f.write(f"  Std:    {np.std(match_distances):.3f}\n")
                f.write(f"  Min:    {np.min(match_distances):.3f}\n")
                f.write(f"  Max:    {np.max(match_distances):.3f}\n")
        print(f"Report saved to: {args.output}")
    
    print("=" * 60)
    
    # Return code based on match quality
    match_quality = stats['match_rate1']  # Percentage of GPU spots matched
    if match_quality >= 90:
        print("✓ Excellent match!")
        return 0
    elif match_quality >= 80:
        print("✓ Good match")
        return 0
    elif match_quality >= 70:
        print("⚠ Moderate match - may need parameter tuning")
        return 1
    else:
        print("⚠ Poor match - significant differences detected")
        return 1


if __name__ == "__main__":
    sys.exit(main())

