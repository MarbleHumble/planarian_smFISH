import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit


def plot_3d_gaussian_fit(img, coord, radius=5, expected_sigma=(1, 1, 1), save_path=None):
    """
    Plot a single 3D spot and its Gaussian fit.

    Args:
        img (np.ndarray): 3D image (Z,Y,X)
        coord (tuple): center of the spot (z,y,x)
        radius (int): half-size of the patch
        expected_sigma (tuple): initial guess for Gaussian (sz,sy,sx)
        save_path (str): path to save plot (PNG)
    """
    zc, yc, xc = coord
    Z, Y, X = img.shape
    z1, z2 = max(0, zc - radius), min(Z, zc + radius + 1)
    y1, y2 = max(0, yc - radius), min(Y, yc + radius + 1)
    x1, x2 = max(0, xc - radius), min(X, xc + radius + 1)

    patch = img[z1:z2, y1:y2, x1:x2]

    zz, yy, xx = np.meshgrid(np.arange(patch.shape[0]),
                             np.arange(patch.shape[1]),
                             np.arange(patch.shape[2]), indexing='ij')

    # 3D Gaussian model
    def gaussian_3d(coords, amp, z0, y0, x0, sz, sy, sx):
        z, y, x = coords
        return amp * np.exp(
            -((z - z0) ** 2 / (2 * sz ** 2) + (y - y0) ** 2 / (2 * sy ** 2) + (x - x0) ** 2 / (2 * sx ** 2))
        ).ravel()

    p0 = [patch.max(), patch.shape[0] / 2, patch.shape[1] / 2, patch.shape[2] / 2, *expected_sigma]
    try:
        popt, _ = curve_fit(gaussian_3d, (zz, yy, xx), patch.ravel(), p0=p0)
        fitted = gaussian_3d((zz, yy, xx), *popt).reshape(patch.shape)
        r2 = 1 - np.sum((patch.ravel() - fitted.ravel()) ** 2) / np.sum((patch.ravel() - patch.mean()) ** 2)
    except Exception:
        fitted = np.zeros_like(patch)
        r2 = 0

    # -----------------------------
    # 2D slices for visualization
    # -----------------------------
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    slices = [
        (patch[patch.shape[0] // 2, :, :], fitted[patch.shape[0] // 2, :, :], "Z Slice"),
        (patch[:, patch.shape[1] // 2, :], fitted[:, patch.shape[1] // 2, :], "Y Slice"),
        (patch[:, :, patch.shape[2] // 2], fitted[:, :, patch.shape[2] // 2], "X Slice"),
    ]
    for i, (raw, fit, title) in enumerate(slices):
        axes[0, i].imshow(raw, cmap="magma")
        axes[0, i].set_title(f"{title} (Raw)")
        axes[1, i].imshow(fit, cmap="magma")
        axes[1, i].set_title(f"{title} (Gaussian Fit)")
    plt.suptitle(f"Spot at {coord}, R²={r2:.2f}")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()