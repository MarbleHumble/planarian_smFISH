#!/usr/bin/env python3
"""
run_server.py
-------------
Server mode entry point for smFISH spot detection pipeline.

Modified to:
    - Print full loaded config
    - Output detected spots at each processing step

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

    print("===== Loaded Config =====")
    for k, v in config.items():
        print(f"{k}: {v}")
    print("==========================\n")

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
    # Detect experiment spots (stepwise)
    # ------------------------------
    print("\n===== Detecting experiment spots =====")
    # Use GPU if available
    use_gpu = config.get("use_gpu", False)
    if use_gpu:
        import torch
        if not torch.cuda.is_available():
            print("WARNING: CUDA unavailable, falling back to CPU.")
            use_gpu = False

    if use_gpu:
        from functions.gpu_smfish import detect_spots_gpu_full, set_max_performance
        set_max_performance()
        print("Using GPU pipeline (Raj 3D LoG + 2D Gaussian)")

        spots_exp, exp_threshold_used, img_log_exp, sum_intensities, radii, good_coords, bad_coords = \
            detect_spots_gpu_full(
                image_np=imread(img_path),
                sigma=tuple(config["kernel_size"]),
                min_distance=tuple(config["minimal_distance"]),
                gaussian_radius=int(config.get("plot_spot_size", 2)),
                gaussian_fit_fraction=float(config.get("gaussian_fit_fraction", 1.0)),
                r2_threshold=float(config.get("r2_threshold", 0.4)),
                random_seed=int(config.get("random_seed", 0)),
                device=config.get("gpu_device", "cuda"),:Wq
                voxel_size=tuple(config.get("voxel_size", (361, 75, 75))),
                min_size_um=float(config.get("radius_for_spots", 200)),
                diagnostic_folder=results_folder if config.get("save_diagnostics", True) else None
            )

        # Stepwise outputs
        print(f"Step 1: After LoG filtering - total spots proposed: {len(spots_exp) + (len(bad_coords) if bad_coords is not None else 0)}")
        if sum_intensities is not None:
            print(f"Step 2: After Raj plateau filtering - spots above intensity threshold: {np.sum(sum_intensities >= exp_threshold_used)}")
        if good_coords is not None:
            print(f"Step 3: After Gaussian fitting - accepted: {len(good_coords)}, rejected: {len(bad_coords)}")

    else:
        # CPU fallback
        from bigfish.stack import log_filter
        from bigfish.detection import detect_spots as bf_detect_spots
        print("Using CPU pipeline (Big-FISH)")

        img_exp = imread(img_path)
        threshold_to_use = config.get("experimentThreshold", None)
        spots_exp, exp_threshold_used = bf_detect_spots(
            images=img_exp,
            threshold=threshold_to_use,
            return_threshold=True,
            voxel_size=config["voxel_size"],
            spot_radius=config["spot_size"],
            log_kernel_size=config["kernel_size"],
            minimum_distance=config["minimal_distance"],
        )
        img_log_exp = log_filter(img_exp, config["kernel_size"])
        good_coords, bad_coords, sum_intensities, radii = None, None, None, None

    # ------------------------------
    # Save outputs
    # ------------------------------
    imwrite(os.path.join(results_folder, "experiment_LoG_filtered.tif"), img_log_exp, photometric="minisblack")
    np.save(os.path.join(results_folder, "experiment_spots.npy"), spots_exp)
    if good_coords is not None:
        np.save(os.path.join(results_folder, "experiment_spots_good.npy"), good_coords)
    if bad_coords is not None:
        np.save(os.path.join(results_folder, "experiment_spots_bad.npy"), bad_coords)
    if sum_intensities is not None:
        np.save(os.path.join(results_folder, "experiment_spot_intensity.npy"), sum_intensities)
    if radii is not None:
        np.save(os.path.join(results_folder, "experiment_spot_radii.npy"), radii)

    print(f"\nPipeline completed. Results saved to: {results_folder}")

if __name__ == "__main__":
    main()

