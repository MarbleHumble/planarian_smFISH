"""
Server mode entry point for smFISH detection pipeline.
Author: Elias Guan
"""

import os
import argparse
import time
import torch
import numpy as np
from tifffile import imwrite

from functions.io_utils import load_config, create_folder_in_same_directory
from functions.spot_detection import detect_spots_from_config
from functions.visualization import plot_spot_example


# -------------------------------------------------
# Argument parsing
# -------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run smFISH spot detection pipeline (server mode)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default=None,
        help="Path to the YAML configuration file"
    )
    return parser.parse_args()


# -------------------------------------------------
# Main pipeline
# -------------------------------------------------
def main():
    t_start_total = time.perf_counter()

    # ------------------------------
    # Step 0 — Parse arguments
    # ------------------------------
    args = parse_arguments()

    # ------------------------------
    # Step 0.5 — GPU detection
    # ------------------------------
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU detected. Using CPU.")

    # ------------------------------
    # Step 1 — Load config.yaml
    # ------------------------------
    t0 = time.perf_counter()
    config_path = args.config or os.path.join(
        os.path.dirname(__file__), "config.yaml"
    )
    config = load_config(config_path)
    t_config = time.perf_counter() - t0

    print(f"Using config: {config_path}")
    print("Loaded config parameters:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("--------------------------------------------------")

    # ------------------------------
    # Step 2 — Results folders
    # ------------------------------
    exp_path = config["smFISHChannelPath"]
    results_folder = create_folder_in_same_directory(exp_path, "results")

    npy_folder = os.path.join(results_folder, "npy")
    tiff_folder = os.path.join(results_folder, "tiff")
    plots_folder = os.path.join(results_folder, "plots")

    for folder in (npy_folder, tiff_folder, plots_folder):
        os.makedirs(folder, exist_ok=True)

    print(f"Results folder: {results_folder}")
    print("--------------------------------------------------")

    # ------------------------------
    # Step 3 — Spot detection
    # ------------------------------
    t0 = time.perf_counter()
    (
        spots_exp,
        threshold_used,
        img_log_exp,
        sum_intensities,
        radii,
        good_coords,
        bad_coords,
    ) = detect_spots_from_config(
        config=config,
        results_folder=results_folder,
    )
    t_detection = time.perf_counter() - t0

    # ------------------------------
    # Step 4 — Reporting
    # ------------------------------
    n_good = len(spots_exp)
    n_bad = len(bad_coords) if bad_coords is not None else 0
    n_total = n_good + n_bad

    print(f"Detected {n_good} good spots")
    print(f"Filtered out {n_bad} bad spots")
    print(f"Total candidate spots: {n_total}")
    print(f"Threshold used: {threshold_used}")

    # ------------------------------
    # Step 5 — Save outputs
    # ------------------------------
    t0 = time.perf_counter()

    np.save(os.path.join(npy_folder, "spots_exp.npy"), spots_exp)

    if sum_intensities is not None:
        np.save(
            os.path.join(npy_folder, "spot_sum_intensity.npy"),
            sum_intensities,
        )

    if radii is not None:
        np.save(
            os.path.join(npy_folder, "spot_radii_zyx.npy"),
            radii,
        )

    imwrite(
        os.path.join(tiff_folder, "smFISH_LoG_filtered.tif"),
        img_log_exp,
        photometric="minisblack",
    )

    t_saving = time.perf_counter() - t0

    # ------------------------------
    # Step 6 — Plot example spots
    # ------------------------------
    t_plotting = 0.0
    if config.get("spotsRadiusDetection", False) and n_good > 0:
        t0 = time.perf_counter()

        radius = int(config.get("plot_spot_size", 2))

        # ---- Gaussian-like spot ----
        coord_good = (
            good_coords[0]
            if good_coords is not None and len(good_coords) > 0
            else spots_exp[0]
        )

        plot_spot_example(
            img=img_log_exp,
            coord=coord_good,
            radius=radius,
            save_path=os.path.join(
                plots_folder, "spot_example_gaussian.png"
            ),
            title="Gaussian-like spot",
        )

        # ---- Non-Gaussian spot ----
        if bad_coords is not None and len(bad_coords) > 0:
            plot_spot_example(
                img=img_log_exp,
                coord=bad_coords[0],
                radius=radius,
                save_path=os.path.join(
                    plots_folder, "spot_example_non_gaussian.png"
                ),
                title="Non-Gaussian spot",
            )

        t_plotting = time.perf_counter() - t0

    # ------------------------------
    # Step 7 — Timing summary
    # ------------------------------
    t_total = time.perf_counter() - t_start_total

    print("\n================= TIMING SUMMARY =================")
    print(f"Config loading     : {t_config:8.2f} s")
    print(f"Spot detection     : {t_detection:8.2f} s")
    print(f"Saving outputs     : {t_saving:8.2f} s")
    print(f"Plotting           : {t_plotting:8.2f} s")
    print("-------------------------------------------------")
    print(f"TOTAL runtime      : {t_total:8.2f} s")
    print("=================================================\n")

    print("Pipeline completed successfully.")
    print(f"Results saved to: {results_folder}")


if __name__ == "__main__":
    main()
