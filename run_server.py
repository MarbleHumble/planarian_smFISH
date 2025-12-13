# run_server.py
"""
Server mode entry point for smFISH detection pipeline.
Author: Elias Guan
"""

import os
import argparse
import torch
import numpy as np
from tifffile import imwrite
from functions.io_utils import load_config, create_folder_in_same_directory
from functions.spot_detection import detect_spots_from_config
from functions.gpu_smfish import plot_spot_example  # plot helper

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
    # Step 0.5: Force GPU usage if available
    # -------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU detected. Using device: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("No GPU detected. Using CPU.")

    # -------------------------------------------------
    # Step 1: Load config.yaml
    # -------------------------------------------------
    config_path = args.config if args.config else os.path.join(os.path.dirname(__file__), "config.yaml")
    print(f"Using config: {config_path}")
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
    print(f"Created subfolders: {npy_folder}, {tiff_folder}, {plots_folder}")
    print("--------------------------------------------------")

    # -------------------------------------------------
    # Step 4: Run spot detection
    # -------------------------------------------------
    spots_exp, threshold_used, img_log_exp = detect_spots_from_config(
        config,
        results_folder=results_folder,
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

    print(f"Experiment spots detected: {len(spots_exp)}")
    print(f"Threshold used: {threshold_used}")

    # -------------------------------------------------
    # Step 6: Plot examples if spotsRadiusDetection=True
    # -------------------------------------------------
    if config.get("spotsRadiusDetection", False) and len(spots_exp) > 0:
        # First Gaussian spot
        plot_spot_example(
            img_log_exp,
            spots_exp[0],
            gaussian_fit=True,
            save_path=os.path.join(plots_folder, "spot_example_gaussian.png")
        )
        print(f"Saved Gaussian spot example: spot_example_gaussian.png")

        # Simulate a discarded/non-Gaussian spot
        if len(spots_exp) > 1:
            bad_spot = spots_exp[1] + 5  # small offset
            plot_spot_example(
                img_log_exp,
                bad_spot,
                gaussian_fit=False,
                save_path=os.path.join(plots_folder, "spot_example_non_gaussian.png")
            )
            print(f"Saved non-Gaussian spot example: spot_example_non_gaussian.png")

    print("--------------------------------------------------")
    print(f"Results saved in folders:\n  NPY: {npy_folder}\n  TIFF: {tiff_folder}\n  Plots: {plots_folder}")
    print("--------------------------------------------------")


if __name__ == "__main__":
    main()
