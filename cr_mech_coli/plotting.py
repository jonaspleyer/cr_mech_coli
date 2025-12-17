import matplotlib.pyplot as plt
from matplotlib import colors

COLOR1 = "#6bd2db"
COLOR2 = "#0ea7b5"
COLOR3 = "#0c457d"
COLOR4 = "#ffbe4f"
COLOR5 = "#e8702a"
COLOR6 = "#a02b08"

cmap = colors.LinearSegmentedColormap.from_list(
    "mymap",
    [
        (0.00, colors.hex2color(COLOR3)),
        (0.25, colors.hex2color(COLOR2)),
        (0.50, colors.hex2color(COLOR1)),
        (0.75, colors.hex2color(COLOR4)),
        (1.00, colors.hex2color(COLOR5)),
    ],
)


def set_mpl_rc_params():
    plt.rcParams.update(
        {
            "font.family": "Courier New",  # monospace font
            "font.size": 20,
            "axes.titlesize": 20,
            "axes.labelsize": 20,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 20,
            "figure.titlesize": 20,
        }
    )


def configure_ax(ax, minor=True):
    ax.grid(True, which="major", linestyle="-", linewidth=0.75, alpha=0.25)
    ax.minorticks_on()
    if minor:
        ax.grid(True, which="minor", linestyle="-", linewidth=0.25, alpha=0.15)
    else:
        ax.grid(False, which="minor")
    ax.set_axisbelow(True)
