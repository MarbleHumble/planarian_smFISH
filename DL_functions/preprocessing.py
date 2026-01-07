import numpy as np
from skimage import filters


def log_filter_image(image_array, kernel_size):
    """
    Apply Laplacian of Gaussian (LoG) filter to a 3D or 2D image.

    Args:
        image_array (np.ndarray): Input image (Z, Y, X) or (Y, X)
        kernel_size (tuple or float): Kernel size for LoG, can be a tuple for 3D or float for 2D

    Returns:
        np.ndarray: LoG filtered image
    """
    if image_array.ndim == 3 and isinstance(kernel_size, tuple):
        # Apply LoG filter slice by slice for simplicity
        filtered = np.zeros_like(image_array, dtype=np.float32)
        for z in range(image_array.shape[0]):
            filtered[z] = filters.gaussian_laplace(image_array[z], sigma=kernel_size[1:])
        return filtered
    elif image_array.ndim == 2 and isinstance(kernel_size, (int, float)):
        return filters.gaussian_laplace(image_array, sigma=kernel_size)
    else:
        raise ValueError("Image dimensions and kernel_size shape do not match")


def normalize_image(image_array):
    """
    Normalize image to 0-1 range.

    Args:
        image_array (np.ndarray): Input image

    Returns:
        np.ndarray: Normalized image
    """
    min_val = np.min(image_array)
    max_val = np.max(image_array)
    if max_val - min_val == 0:
        return np.zeros_like(image_array)
    return (image_array - min_val) / (max_val - min_val)