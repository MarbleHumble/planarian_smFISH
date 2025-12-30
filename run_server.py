#!/usr/bin/env python3
"""
run_server.py
-------------
Server mode entry point for smFISH spot detection pipeline.

This version:
    - ALWAYS uses detect_spots_from_config()
    - Correctly activates gpu_smfish_v2
    - Prevents accidental use of legacy Gaussian pipeline
    - Keeps control-image threshold logic

Author: Elias Guan
"""

import os
import argparse
import yaml
import numpy as np
from scipy.spatial.distance import cdist
from tifffile import imwrite

from functions.spot_detection import (
    detect_spots_from_config,
    control_peak_intensities,
    compute_control_threshold_from_peaks,
)

# ============================================================
# --------------------- COMPARISON ---------------------------
# ============================================================

def compare_with_cpu_results(gpu_results_path, cpu_results_path, tolerance=3.0):
    """
    Compare GPU Big-FISH results with CPU Big-FISH results.
    
    Args:
        gpu_results_path: Path to GPU results .npy file
        cpu_results_path: Path to CPU Big-FISH results .npy file
        tolerance: Maximum distance for matching spots (pixels)
    """
    
    try:
        spots_gpu = np.load(gpu_results_path)
        print(f"\n{'='*60}")
        print("Comparing GPU and CPU Big-FISH Results")
        print(f"{'='*60}")
        print(f"GPU results: {len(spots_gpu)} spots from {gpu_results_path}")
    except Exception as e:
        print(f"\nWARNING: Could not load GPU results: {e}")
        return
    
    try:
        spots_cpu = np.load(cpu_results_path)
        print(f"CPU results: {len(spots_cpu)} spots from {cpu_results_path}\n")
    except FileNotFoundError:
        print(f"CPU results file not found: {cpu_results_path}")
        print("Skipping comparison.\n")
        return
    except Exception as e:
        print(f"WARNING: Could not load CPU results: {e}")
        return
    
    # Match spots using distance-based matching
    print(f"Matching spots with tolerance = {tolerance} pixels...")
    
    if len(spots_gpu) == 0 or len(spots_cpu) == 0:
        print("WARNING: One or both result sets are empty. Cannot compare.")
        return
    
    # Compute pairwise distances
    distances = cdist(spots_gpu, spots_cpu, metric='euclidean')
    
    # Find matches: each spot in GPU matches to closest spot in CPU if within tolerance
    matched_pairs = []
    matched_cpu = set()
    
    for i in range(len(spots_gpu)):
        closest_idx = np.argmin(distances[i, :])
        if distances[i, closest_idx] <= tolerance and closest_idx not in matched_cpu:
            matched_pairs.append((i, closest_idx))
            matched_cpu.add(closest_idx)
    
    # Find unmatched spots
    matched_gpu = set(pair[0] for pair in matched_pairs)
    unmatched_gpu = [i for i in range(len(spots_gpu)) if i not in matched_gpu]
    unmatched_cpu = [i for i in range(len(spots_cpu)) if i not in matched_cpu]
    
    # Compute statistics
    count_diff = len(spots_gpu) - len(spots_cpu)
    count_diff_percent = (count_diff / len(spots_cpu) * 100) if len(spots_cpu) > 0 else 0
    match_rate_gpu = (len(matched_pairs) / len(spots_gpu) * 100) if len(spots_gpu) > 0 else 0
    match_rate_cpu = (len(matched_pairs) / len(spots_cpu) * 100) if len(spots_cpu) > 0 else 0
    
    # Compute match distances
    match_distances = []
    for idx_gpu, idx_cpu in matched_pairs:
        dist = np.linalg.norm(spots_gpu[idx_gpu] - spots_cpu[idx_cpu])
        match_distances.append(dist)
    match_distances = np.array(match_distances)
    
    # Print report
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    print(f"GPU Results:      {len(spots_gpu):>8} spots")
    print(f"CPU Results:      {len(spots_cpu):>8} spots")
    print(f"Difference:       {count_diff:>8} spots ({count_diff_percent:+.2f}%)")
    print()
    print(f"Matched spots:    {len(matched_pairs):>8} pairs")
    print(f"  - Match rate (GPU): {match_rate_gpu:>6.2f}%")
    print(f"  - Match rate (CPU): {match_rate_cpu:>6.2f}%")
    print()
    print(f"Unmatched GPU:    {len(unmatched_gpu):>8} spots")
    print(f"Unmatched CPU:    {len(unmatched_cpu):>8} spots")
    print()
    
    if len(match_distances) > 0:
        print("Match Distance Statistics (pixels):")
        print(f"  Mean:   {np.mean(match_distances):.3f}")
        print(f"  Median: {np.median(match_distances):.3f}")
        print(f"  Std:    {np.std(match_distances):.3f}")
        print(f"  Min:    {np.min(match_distances):.3f}")
        print(f"  Max:    {np.max(match_distances):.3f}")
        print()
    
    # Coordinate range comparison
    print("Coordinate Statistics:")
    print(f"GPU spots - Z: [{spots_gpu[:, 0].min():.1f}, {spots_gpu[:, 0].max():.1f}], "
          f"Y: [{spots_gpu[:, 1].min():.1f}, {spots_gpu[:, 1].max():.1f}], "
          f"X: [{spots_gpu[:, 2].min():.1f}, {spots_gpu[:, 2].max():.1f}]")
    print(f"CPU spots - Z: [{spots_cpu[:, 0].min():.1f}, {spots_cpu[:, 0].max():.1f}], "
          f"Y: [{spots_cpu[:, 1].min():.1f}, {spots_cpu[:, 1].max():.1f}], "
          f"X: [{spots_cpu[:, 2].min():.1f}, {spots_cpu[:, 2].max():.1f}]")
    print()
    print(f"{'='*60}\n")
    
    # Quality assessment
    if match_rate_gpu >= 90:
        print("✓ Excellent match!")
    elif match_rate_gpu >= 80:
        print("✓ Good match")
    elif match_rate_gpu >= 70:
        print("⚠ Moderate match - may need parameter tuning")
    else:
        print("⚠ Poor match - significant differences detected")

# ============================================================
# --------------------------- MAIN ---------------------------
# ============================================================

def main():
    # ------------------------------
    # Parse arguments
    # ------------------------------
    parser = argparse.ArgumentParser(description="smFISH spot detection server")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    args = parser.parse_args()

    # ------------------------------
    # Load YAML config
    # ------------------------------
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print("\n===== Loaded Config =====")
    for k, v in config.items():
        print(f"{k}: {v}")
    print("==========================\n")

    # ------------------------------
    # Paths
    # ------------------------------
    img_path = config["smFISHChannelPath"]
    results_folder = config.get(
        "results_folder",
        os.path.join(os.path.dirname(img_path), "results")
    )
    os.makedirs(results_folder, exist_ok=True)

    # ------------------------------
    # Optional control threshold
    # ------------------------------
    if config.get("controlImage") and config.get("controlPath"):
        print("Computing control-image threshold...")
        _, peak_values, _ = control_peak_intensities(
            config["controlPath"], config, results_folder
        )
        control_threshold = compute_control_threshold_from_peaks(
            peak_values, percentile=99
        )
        config["experimentThreshold"] = control_threshold
        print(f"Control-derived threshold: {control_threshold}")

    # ------------------------------
    # Detect experiment spots
    # ------------------------------
    print("\n===== Detecting experiment spots =====")

    (
        spots_exp,
        exp_threshold_used,
        img_log_exp,
        sum_intensities,
        radii,
        good_coords,
        bad_coords,
    ) = detect_spots_from_config(
        config,
        img_path=img_path,
        results_folder=results_folder,
    )

    # ------------------------------
    # Save outputs
    # ------------------------------
    np.save(
        os.path.join(results_folder, "experiment_spots.npy"),
        spots_exp
    )

    if img_log_exp is not None:
        imwrite(
            os.path.join(results_folder, "experiment_LoG_filtered.tif"),
            img_log_exp,
            photometric="minisblack",
        )

    if good_coords is not None:
        np.save(
            os.path.join(results_folder, "experiment_spots_good.npy"),
            good_coords
        )

    if bad_coords is not None:
        np.save(
            os.path.join(results_folder, "experiment_spots_bad.npy"),
            bad_coords
        )

    if sum_intensities is not None:
        np.save(
            os.path.join(results_folder, "experiment_spot_intensity.npy"),
            sum_intensities
        )

    if radii is not None:
        np.save(
            os.path.join(results_folder, "experiment_spot_radii.npy"),
            radii
        )

    print(
        f"\nPipeline completed successfully.\n"
        f"Detected {len(spots_exp)} experiment spots.\n"
        f"Results saved to: {results_folder}"
    )
    
    # ------------------------------
    # Compare with CPU Big-FISH results (if available)
    # ------------------------------
    cpu_results_path = config.get("cpu_bigfish_results_path", None)
    if cpu_results_path:
        gpu_results_path = os.path.join(results_folder, "experiment_spots.npy")
        tolerance = config.get("comparison_tolerance", 3.0)
        compare_with_cpu_results(gpu_results_path, cpu_results_path, tolerance=tolerance)
    else:
        print("\n(CPU Big-FISH results path not specified in config, skipping comparison)")


# ============================================================
# -------------------------- ENTRY ---------------------------
# ============================================================

if __name__ == "__main__":
    main()

