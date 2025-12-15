
"""
Spot detection functions for smFISH pipeline (server mode)
Author: Elias Guan
"""

from tifffile import imread, imwrite
from bigfish.stack import log_filter
from bigfish.detection import detect_spots as bf_detect_spots
import numpy as np
import os


def detect_control_spots(control_path, config, results_folder):
    """
    Run BigFISH spot detection on control image with threshold=0
    to get all candidate spots. This is used to set an appropriate
    threshold for the real experiment image.

    Args:
        control_path (str): Path to control image
        config (dict): Configuration parameters
        results_folder (str): Folder to save results

    Returns:
        np.ndarray: All candidate spots detected on the control image
    """
    # Load image
    img = imread(control_path)
    print(f"Loaded control image: {img.shape}")

    # Apply LoG filter
    kernel_size = config["kernel_size"]
    img_log = log_filter(img, kernel_size)
    imwrite(os.path.join(results_folder, "control_LoG_filtered.tif"), img_log)

    # Detect all spots with threshold=0
    spots_all, _ = bf_detect_spots(
        images=img,
        threshold=0,
        return_threshold=True,
        voxel_size=config["voxel_size"],
        spot_radius=config["spot_size"],
        log_kernel_size=kernel_size,
        minimum_distance=config["minimal_distance"]
    )

    print(f"Detected {len(spots_all)} candidate spots on control image (threshold=0).")
    return spots_all


def find_spots_around(coordinate, array, max_iterations=10):
    """
    Flood-fill to find all voxels belonging to a 3D spot starting from a coordinate.
    Uses 6-connectivity (faces only) and stops at zero intensity or max_iterations.

    Args:
        coordinate (tuple/list/array): (z, y, x) starting voxel
        array (np.ndarray): 3D LoG-filtered image
        max_iterations (int): Maximum iterations to avoid infinite loops

    Returns:
        np.ndarray: Array of voxel coordinates belonging to the spot
    """
    # 6-connectivity vectors
    vectors = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ], dtype=int)

    outer_edge = [np.array(coordinate, dtype=np.int16)]
    spots_collection = [np.array(coordinate, dtype=np.int16)]
    shape = array.shape

    for _ in range(max_iterations):
        new_outer_edge = []
        for item in outer_edge:
            for vec in vectors:
                neighbor = item + vec
                # Skip out-of-bounds neighbors
                if np.any(neighbor < 0) or np.any(neighbor >= shape):
                    continue
                # Skip already visited
                if any(np.array_equal(neighbor, v) for v in spots_collection):
                    continue
                # Add neighbor if intensity > 0 and decreasing from center
                if array[tuple(neighbor)] > 0 and array[tuple(item)] >= array[tuple(neighbor)]:
                    spots_collection.append(neighbor)
                    new_outer_edge.append(neighbor)
        if len(new_outer_edge) == 0:
            break
        outer_edge = new_outer_edge

    return np.array(spots_collection, dtype=np.int16)





def compute_control_spot_intensities(control_path, config, results_folder):
    if not os.path.exists(control_path):
        raise FileNotFoundError(f"Control image not found: {control_path}")

    img = imread(control_path)
    kernel_size = config["kernel_size"]
    img_log = log_filter(img, kernel_size)
    imwrite(os.path.join(results_folder, "control_LoG_filtered.tif"), img_log)

    spots_all, _ = bf_detect_spots(
        images=img,
        threshold=0,
        return_threshold=True,
        voxel_size=config["voxel_size"],
        spot_radius=config["spot_size"],
        log_kernel_size=kernel_size,
        minimum_distance=config["minimal_distance"]
    )

    print(f"Detected {len(spots_all)} candidate spots at threshold=0.")

    # YOUR LoG-integrated intensity
    total_intensities = []
    # NEW: LoG peak (center voxel) intensity
    peak_intensities = []

    for spot in spots_all:
        z, y, x = spot[0], spot[1], spot[2]

        coords = find_spots_around((z, y, x), img_log)
        total_intensities.append(np.sum(img_log[tuple(coords.T)]))

        peak_intensities.append(img_log[z, y, x])   # <--- ADD THIS

    return (
        spots_all,
        np.array(total_intensities),
        np.array(peak_intensities),  # <--- NEW RETURN VALUE
        img_log
    )

def control_peak_intensities(control_path, config, results_folder):
    """
    Detect all candidate centroids in the control image (threshold=0),
    compute the LoG image, and return the LoG peak value at each centroid.

    Returns:
        spots_all: ndarray of centroids (N,3)
        peak_values: 1D ndarray of LoG peak values (N,)
        img_log: the LoG-filtered image (for inspection / saving)
    """
    if not os.path.exists(control_path):
        raise FileNotFoundError(f"Control image not found: {control_path}")

    # load and LoG-filter
    img = imread(control_path)
    kernel_size = config["kernel_size"]
    img_log = log_filter(img, kernel_size)
    imwrite(os.path.join(results_folder, "control_LoG_filtered.tif"), img_log)

    # detect all local maxima (threshold=0)
    spots_all, _ = bf_detect_spots(
        images=img,
        threshold=0,
        return_threshold=True,
        voxel_size=config["voxel_size"],
        spot_radius=config["spot_size"],
        log_kernel_size=kernel_size,
        minimum_distance=config["minimal_distance"]
    )

    # get LoG peak at centroid for each detected spot
    peak_values = []
    for s in spots_all:
        z, y, x = int(s[0]), int(s[1]), int(s[2])
        peak_values.append(float(img_log[z, y, x]))

    return spots_all, np.array(peak_values), img_log

def compute_control_threshold(total_intensities, peak_intensities, percentile=0.99):
    """
    Args:
        total_intensities: array of your flood-filled LoG intensities
        peak_intensities: array of LoG peak values at spot centroid
        percentile: 0.99 = use the 99th percentile spot
    Returns:
        threshold (float): LoG threshold for BigFISH
        selected_index (int): index of chosen spot
    """

    # sort by your custom total intensity
    sorted_indices = np.argsort(total_intensities)
    idx = sorted_indices[int(len(total_intensities) * percentile)]

    # threshold = LoG peak intensity at that spot
    threshold = peak_intensities[idx]

    print(f"Control threshold (LoG peak at {percentile*100}%): {threshold}")

    return threshold, idx

# functions/spot_detection.py

def compute_control_threshold_from_peaks(peak_values, percentile=99):
    """
    Use the percentile (0-100) of centroid LoG peaks as the control threshold.
    """
    if len(peak_values) == 0:
        return 0.0  # no peaks -> fallback to 0
    thr = float(np.percentile(peak_values, percentile))
    return thr

import os
import numpy as np
from tifffile import imread, imwrite

def detect_spots_from_config(config, img_path=None, results_folder=None):
    """
    smFISH spot detection wrapper supporting GPU (Raj-style 3D LoG + 2D Gaussian)
    and CPU fallback using Big-FISH.

    Returns:
        spots_exp (np.ndarray): detected spot coordinates
        exp_threshold_used (float or None): intensity threshold applied
        img_log_exp (np.ndarray): LoG-filtered image
        sum_intensities (np.ndarray or None): 3D integrated intensities
        radii (np.ndarray or None): spot radii (z,y,x)
        good_coords (np.ndarray or None): spots passing Gaussian fit
        bad_coords (np.ndarray or None): spots failing Gaussian fit
    """
    # ------------------------------
    # Paths
    # ------------------------------
    if img_path is None:
        img_path = config["smFISHChannelPath"]
    if results_folder is None:
        results_folder = os.path.join(os.path.dirname(img_path), "results")
    os.makedirs(results_folder, exist_ok=True)

    # ------------------------------
    # Load experiment image
    # ------------------------------
    img_exp = imread(img_path)
    print(f"Loaded experiment image: {img_exp.shape}")

    compute_radii = bool(config.get("spotsRadiusDetection", False))

    # ------------------------------
    # GPU backend
    # ------------------------------
    use_gpu = bool(config.get("use_gpu", False))
    if use_gpu:
        import torch
        if not torch.cuda.is_available():
            print("WARNING: CUDA unavailable, falling back to CPU.")
            use_gpu = False

    if use_gpu:
        from functions.gpu_smfish import detect_spots_gpu_full, set_max_performance
        set_max_performance()
        print("smFISH backend: GPU (Raj + 2D Gaussian + Size Filter)")

        spots_exp, exp_threshold_used, img_log_exp, sum_intensities, radii, good_coords, bad_coords = \
            detect_spots_gpu_full(
                image_np=img_exp,
                sigma=tuple(config["kernel_size"]),
                min_distance=tuple(config["minimal_distance"]),
                gaussian_radius=int(config.get("plot_spot_size", 2)),
                gaussian_fit_fraction=float(config.get("gaussian_fit_fraction", 1.0)),
                r2_threshold=float(config.get("r2_threshold", 0.4)),
                random_seed=int(config.get("random_seed", 0)),
                device=config.get("gpu_device", "cuda"),
                voxel_size=tuple(config.get("voxel_size", (361, 75, 75))),
                min_size_um=float(config.get("radius_for_spots", 200)),
                diagnostic_folder=results_folder if config.get("save_diagnostics", True) else None
            )

    # ------------------------------
    # CPU fallback (Big-FISH)
    # ------------------------------
    else:
        from bigfish.stack import log_filter
        from bigfish.detection import detect_spots as bf_detect_spots
        print("smFISH backend: Big-FISH CPU")

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

        # Optional radii computation
        if compute_radii and len(spots_exp) > 0:
            from .spot_detection import find_spots_around
            sum_intensities, radii = [], []
            for coord in spots_exp:
                voxels = find_spots_around(coord, img_log_exp)
                sum_intensities.append(np.sum(img_exp[tuple(voxels.T)]))
                radii.append([
                    voxels[:, 0].ptp() + 1,
                    voxels[:, 1].ptp() + 1,
                    voxels[:, 2].ptp() + 1
                ])
            sum_intensities = np.asarray(sum_intensities)
            radii = np.asarray(radii)
        good_coords, bad_coords = None, None

    # ------------------------------
    # Reporting
    # ------------------------------
    print(f"Detected {len(spots_exp)} experiment spots")

    # ------------------------------
    # Save outputs
    # ------------------------------
    imwrite(os.path.join(results_folder, "experiment_LoG_filtered.tif"), img_log_exp, photometric="minisblack")
    np.save(os.path.join(results_folder, "experiment_spots.npy"), spots_exp)

    if compute_radii and sum_intensities is not None:
        np.save(os.path.join(results_folder, "experiment_spot_intensity.npy"), sum_intensities)
        np.save(os.path.join(results_folder, "experiment_spot_radii.npy"), radii)

    print("Saved experiment results.")

    return spots_exp, exp_threshold_used, img_log_exp, sum_intensities, radii, good_coords, bad_coords




