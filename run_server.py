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
import matplotlib.pyplot as plt

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
# Optional: Plot Raj-style elbow / plateau curve
# -------------------------------------------------
def plot_elbow_curve(intensities, save_path):
    sorted_int = np.sort(intensities)[::-1]
    plt.figure()
    plt.plot(sorted_int, '-o', markersize=2)
    plt.xlabel("Spot index (sorted)")
    plt.ylabel("Integrated intensity")
    plt.title("Raj-style intensity elbow / plateau curve")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    t_start_total = time.perf_counter()

    # Step 0 — Parse arguments
    args = parse_arguments()

    # Step 0.5 — GPU detection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU detected: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("No GPU detected. Using CPU.")

    # Step 1 — Load config.yaml
    t0 = time.perf_counter()
    config_path = args.config or os.path.join(os.path.dirname(__file__), "config.yaml")
    print(f"Using config: {config_path}")
    config = load_config(config_path)
    t_config = time.perf_counter() - t0

    print("\nLoaded config parameters:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("--------------------------------------------------")

    # Step 2 — Create main results folder
    t0 = time.perf_counter()
    exp_path = config.get("smFISHChannelPath")
    results_folder = create_folder_in_same_directory(exp_path, "results")
    t_results_folder = time.perf_counter() - t0
    print(f"Main results folder: {results_folder}")

    # Step 3 — Create subfolders
    t0 = time.perf_counter()
    npy_folder = os.path.join(results_folder, "npy")
    tiff_folder = os.path.join(results_folder, "tiff")
    plots_folder = os.path.join(results_folder, "plots")
    for folder in [npy_folder, tiff_folder, plots_folder]:
        os.makedirs(folder, exist_ok=True)
    t_subfolders = time.perf_counter() - t0
    print(f"Created subfolders:\n  NPY  : {npy_folder}\n  TIFF : {tiff_folder}\n  Plots: {plots_folder}")
    print("--------------------------------------------------")

    # Step 4 — Spot detection (MAIN COMPUTE)
    t0 = time.perf_counter()
    (
        spots_exp,
        threshold_used,
        img_log_exp,
        sum_intensities,
        radii,
        good_c,
        bad_c
    ) = detect_spots_from_config(
        config,
        results_folder=results_folder,
    )
    t_detection = time.perf_counter()

    # Step 4b — Optional: elbow / plateau curve
    if sum_intensities is not None:
        plot_elbow_curve(sum_intensities, os.path.join(plots_folder, "elbow_curve.png"))

    # Count filtered bad spots
    n_total = (len(spots_exp) + len(bad_c)) if bad_c is not None else len(spots_exp)
    n_bad = len(bad_c) if bad_c is not None else 0
    print(f"Detected {len(spots_exp)} good spots")
    print(f"Filtered out {n_bad} bad spots")
    print(f"Total candidate spots: {n_total}")
    print(f"Threshold used: {threshold_used}")

    # Step 5 — Save outputs
    t0 = time.perf_counter()
    np.save(os.path.join(npy_folder, "spots_exp.npy"), spots_exp)
    if sum_intensities is not None:
        np.save(os.path.join(npy_folder, "spot_sum_intensity.npy"), sum_intensities)
    if radii is not None:
        np.save(os.path.join(npy_folder, "spot_radii_zyx.npy"), radii)
    imwrite(os.path.join(tiff_folder, "smFISH_LoG_filtered.tif"), img_log_exp, photometric="minisblack")
    t_saving = time.perf_counter() - t0

    # Step 6 — Plot examples
    t_plotting = 0.0
    if config.get("spotsRadiusDetection", False) and len(spots_exp) > 0:
        t0 = time.perf_counter()
        plot_spot_example(
            img_log_exp,
            good_c[0] if good_c is not None else spots_exp[0],
            gaussian_fit=True,
            save_path=os.path.join(plots_folder, "spot_example_gaussian.png"),
            radius=int(config.get("plot_spot_size", 2)),
            title="Gaussian-like spot"
        )
        if bad_c is not None and len(bad_c) > 0:
            plot_spot_example(
                img_log_exp,
                bad_c[0],
                gaussian_fit=False,
                save_path=os.path.join(plots_folder, "spot_example_non_gaussian.png"),
                radius=int(config.get("plot_spot_size", 2)),
                title="Non-Gaussian spot"
            )
        t_plotting = time.perf_counter() - t0

    # Step 7 — Timing summary
    t_total = time.perf_counter() - t_start_total
    print("\n================= TIMING SUMMARY =================")
    print(f"Config loading        : {t_config:8.2f} s")
    print(f"Results folder        : {t_results_folder:8.2f} s")
    print(f"Subfolder creation    : {t_subfolders:8.2f} s")
    print(f"Spot detection        : {t_detection:8.2f} s")
    print(f"Saving outputs        : {t_saving:8.2f} s")
    print(f"Plotting              : {t_plotting:8.2f} s")
    print("-------------------------------------------------")
    print(f"TOTAL runtime         : {t_total:8.2f} s")
    print("=================================================\n")
    print("Pipeline completed successfully.")
    print(f"Results saved to: {results_folder}")


if __name__ == "__main__":
    main()

