# run_server.py
"""
Server mode entry point for smFISH detection pipeline.
Author: Elias Guan
"""


import os
import argparse
from functions.io_utils import load_config, create_folder_in_same_directory
from functions.spot_detection import detect_spots_from_config
import numpy as np
from tifffile import imwrite


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


def main():
    # -------------------------------------------------
    # Step 0: Parse arguments
    # -------------------------------------------------
    args = parse_arguments()

    # -------------------------------------------------
    # Step 1: Load config.yaml
    # -------------------------------------------------
    if args.config is None:
        # default = use config.yaml in same directory as script
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        print(f"No config provided. Using default: {config_path}")
    else:
        config_path = args.config
        print(f"Using config provided via command line: {config_path}")

    config = load_config(config_path)

    print("\nLoaded config parameters:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("--------------------------------------------------")

    # -------------------------------------------------
    # Step 2: Create main results folder
    # -------------------------------------------------
    exp_path = config.get("smFISHChannelPath")
    results_folder = create_folder_in_same_directory(exp_path, "results")
    print(f"Main results folder: {results_folder}")

    # -------------------------------------------------
    # Step 3: Create subfolders inside results/
    # -------------------------------------------------
    npy_folder = os.path.join(results_folder, "npy")
    tiff_folder = os.path.join(results_folder, "tiff")
    plots_folder = os.path.join(results_folder, "plots")

    for p in [npy_folder, tiff_folder, plots_folder]:
        os.makedirs(p, exist_ok=True)

    print("Created subfolders:")
    print(f"  {npy_folder}")
    print(f"  {tiff_folder}")
    print(f"  {plots_folder}")
    print("--------------------------------------------------")

    # -------------------------------------------------
    # Step 4: Run spot detection
    # -------------------------------------------------
    spots_exp, threshold_used, img_log_exp = detect_spots_from_config(
        config,
        results_folder=results_folder
    )

    # -------------------------------------------------
    # Step 5: Save outputs
    # -------------------------------------------------
    np.save(os.path.join(npy_folder, "spots_exp.npy"), spots_exp)

    imwrite(
        os.path.join(tiff_folder, "smFISH_LoG_filtered.tif"),
        img_log_exp,
        photometric="minisblack"
    )

    print("\n--------------------------------------------------")
    print(f"Experiment spots detected: {len(spots_exp)}")
    print(f"Threshold used: {threshold_used}")
    print("Saved results:")
    print(f"  NPY folder:   {npy_folder}")
    print(f"  TIFF folder:  {tiff_folder}")
    print(f"  Plots folder: {plots_folder}")
    print("--------------------------------------------------")


if __name__ == "__main__":
    main()
