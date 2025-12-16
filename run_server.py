#!/usr/bin/env python3
"""
run_server.py
-------------
Server mode entry point for smFISH spot detection pipeline.

Supports:
    - GPU (Raj-style 3D LoG + 2D Gaussian)
    - CPU fallback using Big-FISH
    - Optional control-image thresholding
    - Optional radii / Gaussian validation
    - Flexible config-driven settings

Author: Elias Guan
"""

import os
import argparse
import yaml
from tifffile import imread, imwrite
import numpy as np

from functions.spot_detection import detect_spots_from_config
from functions.spot_detection import control_peak_intensities, compute_control_threshold_from_peaks

def main():
    # ------------------------------
    # Parse arguments
    # ------------------------------
    parser = argparse.ArgumentParser(description="smFISH spot detection server")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML configuration file")
    args = parser.parse_args()

    # ------------------------------
    # Load YAML config
    # ------------------------------
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    img_path = config.get("smFISHChannelPath")
    results_folder = config.get("results_folder", os.path.join(os.path.dirname(img_path), "results"))
    os.makedirs(results_folder, exist_ok=True)

    # ------------------------------
    # Optional control threshold
    # ------------------------------
    control_threshold = None
    if config.get("controlImage") and config.get("controlPath"):
        print("Computing control-image threshold...")
        _, peak_values, _ = control_peak_intensities(config["controlPath"], config, results_folder)
        control_threshold = compute_control_threshold_from_peaks(peak_values, percentile=99)
        print(f"Control-derived threshold: {control_threshold}")
        config["experimentThreshold"] = control_threshold

    # ------------------------------
    # Detect experiment spots
    # ------------------------------
    spots_exp, exp_threshold_used, img_log_exp, sum_intensities, radii, good_coords, bad_coords = \
        detect_spots_from_config(config, img_path=img_path, results_folder=results_folder)

    # ------------------------------
    # Reporting
    # ------------------------------
    print("--------------------------------------------------")
    print(f"Detected {len(spots_exp)} experiment spots")
    print(f"Threshold used: {exp_threshold_used}")
    if good_coords is not None:
        print(f"Gaussian-fit accepted: {len(good_coords)} spots")
    if bad_coords is not None:
        print(f"Gaussian-fit rejected: {len(bad_coords)} spots")
    print("--------------------------------------------------")

    # ------------------------------
    # Save outputs
    # ------------------------------
    imwrite(os.path.join(results_folder, "experiment_LoG_filtered.tif"), img_log_exp, photometric="minisblack")
    np.save(os.path.join(results_folder, "experiment_spots.npy"), spots_exp)

    if sum_intensities is not None:
        np.save(os.path.join(results_folder, "experiment_spot_intensity.npy"), sum_intensities)
    if radii is not None:
        np.save(os.path.join(results_folder, "experiment_spot_radii.npy"), radii)
    if good_coords is not None:
        np.save(os.path.join(results_folder, "experiment_spots_good.npy"), good_coords)
    if bad_coords is not None:
        np.save(os.path.join(results_folder, "experiment_spots_bad.npy"), bad_coords)

    print(f"Results saved to: {results_folder}")
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
