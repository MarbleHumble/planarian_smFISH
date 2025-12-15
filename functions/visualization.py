import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def plot_spot_example(
    img,
    coord,
    radius=3,
    voxel_size=(1.0, 1.0, 1.0),
    gaussian_fit=True,
    save_path="spot_example.png",
    title=None,
):
    """
    Visualize a single smFISH spot in 2D (XY).

    Args:
        img (np.ndarray): 3D image (Z, Y, X)
        coord (array-like): (z, y, x) voxel coordinate
        radius (int): Half-size of crop in Y/X (Z ignored)
        voxel_size (tuple): Voxel size (Z, Y, X) in microns
        gaussian_fit (bool): Whether this spot passed Gaussian fitting
        save_path (str): Output image path
        title (str or None): Optional custom title
    """
    z, y, x = map(int, coord)
    Z, Y, X = img.shape

    y1, y2 = max(0, y - radius), min(Y, y + radius + 1)
    x1, x2 = max(0, x - radius), min(X, x + radius + 1)

    sub_xy = img[z, y1:y2, x1:x2]

    if sub_xy.size == 0:
        print(f"WARNING: Empty crop for spot {coord}")
        return

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.imshow(sub_xy, cmap="hot", origin="lower")
    ax.axis("off")

    # Overlay a circle for approximate spot size if Gaussian fit
    if gaussian_fit:
        # radius in pixels assuming voxel_size
        radius_pix = radius
        circ = Circle(
            ((x2 - x1) / 2, (y2 - y1) / 2),
            radius_pix,
            edgecolor="cyan",
            facecolor="none",
            lw=1.5,
        )
        ax.add_patch(circ)

    if title is None:
        fit_label = "Gaussian" if gaussian_fit else "Non-Gaussian"
        title = f"{fit_label} spot @ {tuple(coord)}"

    ax.set_title(title, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

