import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import cr_mech_coli as crm
import skimage as sk


def get_skeleton(submask, color):
    return sk.morphology.skeletonize(np.all(submask == color, axis=2), method="lee")


if __name__ == "__main__":
    mask = np.loadtxt("data/crm_fit/0001/masks/001042-markers.csv", delimiter=",")

    # Create new colors for mask
    new_colors = {int(i): crm.counter_to_color(int(i)) for i in np.unique(mask)}
    new_mask = np.zeros((*mask.shape, 3), int)
    for c in np.unique(mask):
        new_mask[mask == c] = new_colors[int(c)]
    mask = new_mask

    # Define limits of plot
    fig, ax = plt.subplots()
    ax.set_xlim((0, mask.shape[1]))
    ax.set_ylim((mask.shape[0], 0))
    ax.set_axis_off()
    ax.imshow(mask)

    # Pick one cell
    color = np.array(new_colors[2])
    x, y = np.where(np.all(mask == color, axis=2))
    xmin = np.min(x)
    xmax = np.max(x)
    dx = xmax - xmin
    ymin = np.min(y)
    ymax = np.max(y)
    dy = ymax - ymin

    rect = Rectangle(
        (float(ymin), float(xmin)),
        float(dy),
        float(dx),
        color="white",
        fill=False,
        alpha=0.3,
    )
    ax.add_patch(rect)

    submask = np.copy(mask[xmin:xmax, ymin:ymax, :])
    magnifier = 6
    dt = magnifier * dy
    tmin = np.intp(200)
    tmax = tmin + dt
    ds = magnifier * dx
    smin = mask.shape[0] - 240 - magnifier * dx
    smax = smin + ds
    # First draw lines which mimick zooming glass
    ax.plot([tmin, ymin], [smin, xmin], color="w", alpha=0.3)
    ax.plot([tmin, ymin], [smax, xmax], color="w", alpha=0.3)
    ax.plot([tmax, ymax], [smin, xmin], color="w", alpha=0.3)
    ax.plot([tmax, ymax], [smax, xmax], color="w", alpha=0.3)

    # Now draw rectangle in which we are zooming in
    rect2 = Rectangle(
        (float(tmin - 1), float(smin - 2)),
        float(dt + 2),
        float(ds + 2),
        color="white",
        fill=False,
        alpha=0.3,
        zorder=21,
    )
    ax.add_patch(rect2)
    # Thin the mask; obtain the skeleton
    skeleton = get_skeleton(submask, color)
    submask[skeleton != 0] = [255, 255, 255]
    # Now draw magnified subsection of image
    ax.imshow(
        submask,
        extent=(float(tmin), float(tmax), float(smax), float(smin)),
        zorder=20,
    )

    fig.tight_layout()
    fig.savefig(
        "docs/source/_static/fitting-methods/algorithm/mask-zoom.png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)

    pos, _, _ = crm.extract_positions(mask)
    submask = np.copy(mask[xmin:xmax, ymin:ymax, :])

    fig, ax = plt.subplots()
    ax.set_xlim((0, submask.shape[1]))
    ax.set_ylim((submask.shape[0], 0))
    ax.set_axis_off()
    ax.imshow(submask)

    colors = list(
        filter(
            lambda x: np.all(x != [0, 0, 0]),
            np.unique(submask.reshape((-1, 3)), axis=0),
        )
    )
    c1 = colors[0]
    c2 = colors[1]
    skeleton1 = get_skeleton(mask, c1)
    skeleton2 = get_skeleton(mask, c2)
    for p in pos:
        p1 = p[:, 0] - xmin - dx / 2
        p2 = p[:, 1] - ymin - dy / 2
        q1 = -(-p1 + p2) * (-1) + p1
        q2 = -(-p1 + p2) * (+1) + p2
        ax.plot(q1 + dx / 2, q2 + dy / 2, color="white", marker="+", markersize=15)

    fig.tight_layout()
    fig.savefig(
        "docs/source/_static/fitting-methods/algorithm/interpolate-positions.png",
        bbox_inches="tight",
        pad_inches=0,
    )
