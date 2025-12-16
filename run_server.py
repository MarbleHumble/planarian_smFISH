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
from tifffile import imwrite

from functions.spot_detection import (
    detect_spots_from_config,
    control_peak_intensities,
    compute_control_threshold_from_peaks,
)

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


# ============================================================
# -------------------------- ENTRY ---------------------------
# ============================================================

if __name__ == "__main__":
    main()

