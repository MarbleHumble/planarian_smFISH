import numpy as np
import matplotlib.pyplot as plt


def plot_spot_example(
    img,
    coord,
    radius=3,
    gaussian_fit=True,
    save_path="spot_example.png",
    title=None,
):
    """
    Visualize a single 3D smFISH spot.

    Args:
        img (np.ndarray): 3D image (Z, Y, X)
        coord (array-like): (z, y, x) voxel coordinate
        radius (int): Half-size of crop in Y/X (Z uses radius//2)
        gaussian_fit (bool): Whether this spot passed Gaussian fitting
        save_path (str): Output image path
        title (str or None): Optional custom title
    """
    z, y, x = map(int, coord)
    Z, Y, X = img.shape

    rz = max(1, radius // 2)

    z1, z2 = max(0, z - rz), min(Z, z + rz + 1)
    y1, y2 = max(0, y - radius), min(Y, y + radius + 1)
    x1, x2 = max(0, x - radius), min(X, x + radius + 1)

    sub = img[z1:z2, y1:y2, x1:x2]

    if sub.size == 0:
        print(f"WARNING: Empty crop for spot {coord}")
        return

    # Projections
    mip_xy = sub.max(axis=0)
    mid_z = sub.shape[0] // 2
    mid_y = sub.shape[1] // 2
    mid_x = sub.shape[2] // 2

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    axes[0].imshow(mip_xy, cmap="hot")
    axes[0].set_title("XY (Z MIP)")

    axes[1].imshow(sub[mid_z, :, :], cmap="hot")
    axes[1].set_title("XY (center Z)")

    axes[2].imshow(sub[:, mid_y, :], cmap="hot")
    axes[2].set_title("XZ")

    for ax in axes:
        ax.axis("off")

    if title is None:
        fit_label = "Gaussian" if gaussian_fit else "Non-Gaussian"
        title = f"{fit_label} spot @ {tuple(coord)}"

    plt.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
