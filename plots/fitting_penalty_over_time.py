import cr_mech_coli as crm
import matplotlib.pyplot as plt
from pathlib import Path
import time
import multiprocessing as mp
import scipy as sp
import numpy as np
import argparse
import matplotlib as mpl
from PIL import Image

from fitting_extract_positions import create_simulation_result


def render_single_mask(n_iter: int, cell_container, domain_size, render_settings):
    cell_container = crm.CellContainer.deserialize(cell_container)
    cells_at_iter = cell_container.get_cells_at_iteration(n_iter)
    colors = cell_container.cell_to_color
    res = crm.render_mask(cells_at_iter, colors, domain_size, render_settings)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--n-vertices",
        type=int,
        default=8,
    )
    pyargs = parser.parse_args()
    config, cell_container = create_simulation_result(pyargs.n_vertices)
    iterations = cell_container.get_all_iterations()

    interval = time.time()
    pool = mp.Pool()

    rs = crm.RenderSettings()
    args = [(i, cell_container.serialize(), config.domain_size, rs) for i in iterations]
    masks = pool.starmap(render_single_mask, args)
    print(f"{time.time() - interval:8.4} Calculated Masks:")
    interval = time.time()

    save_interval = config.t_max / (config.n_saves + 1)
    penalties_area_diff = [
        crm.penalty_area_diff(masks[i - 1], masks[i]) / save_interval
        for i in range(1, len(iterations))
    ]
    print(f"{time.time() - interval:8.4} Calculated Penalties without parents:")
    interval = time.time()

    penalties_parents = [
        crm.penalty_area_diff_account_parents(
            masks[i - 1],
            masks[i],
            cell_container.color_to_cell,
            cell_container.parent_map,
            0,
        )
        / save_interval
        for i in range(1, len(iterations))
    ]
    print(f"{time.time() - interval:8.4} Calculated Penalties with parents:")
    interval = time.time()

    n_cells = [len(cell_container.get_cells_at_iteration(i)) for i in iterations]
    x = np.array([i * save_interval for i in range(len(iterations))])

    # Fit exponential function to penalties with parents
    def exponential(x, A, growth):
        return A * np.exp(growth * x)

    popt, pcov = sp.optimize.curve_fit(
        exponential,
        x[1:],
        penalties_parents,
        p0=(0.1, np.log(penalties_parents[-1] / penalties_parents[0]) / (x[-1] - x[0])),
    )

    crm.plotting.set_mpl_rc_params()
    fig = plt.figure(layout="constrained", figsize=(24, 18))
    gs = mpl.gridspec.GridSpec(
        2,
        4,
        wspace=0.01,
        hspace=0.01,
        height_ratios=(1, 2),
        left=0,
        right=1,
        bottom=0,
        top=1,
        figure=fig,
    )
    subfigs = []
    for i in range(4):
        subfigs.append(fig.add_subfigure(gs[0, i]))
    subfigs.append(fig.add_subfigure(gs[1, :2]))
    subfigs.append(fig.add_subfigure(gs[1, 2:]))

    axs = []

    def write_text(ax, label):
        ax.text(
            0.03,
            0.97,
            label,
            fontsize=40,
            fontweight="semibold",
            fontfamily="serif",
            va="top",
            horizontalalignment="left",
            transform=ax.transAxes,
            color="white" if k <= 4 else "k",
        )

    labels = ["A", "B", "C", "D", "E", "F"]
    for k, (label, sf) in enumerate(zip(labels, subfigs[:5]), 1):
        ax = sf.subplots(gridspec_kw={"bottom": 0, "top": 1, "left": 0, "right": 1})
        axs.append(ax)
        write_text(ax, label)
        ax.set_axis_off()
        if k <= 4:
            img = Image.open(
                f"docs/source/_static/fitting-methods/progressions-{k}.png"
            )
            ax.imshow(img, cmap=None if k <= 2 else "Grays_r")
        elif k == 5:
            img = Image.open(
                "docs/source/_static/fitting-methods/extract_positionsdivision-comparison.png"
            )
            ax.imshow(img)

    sf1 = subfigs[-1]
    ax1 = sf1.subplots()
    write_text(ax1, labels[-1])
    crm.plotting.configure_ax(ax1)
    ax1.plot(
        x[1:],
        penalties_area_diff,
        label="Area Difference",
        color=crm.plotting.COLOR3,
    )
    ax1.plot(
        x[1:],
        penalties_parents,
        label="with Parents",
        linestyle="--",
        color=crm.plotting.COLOR2,
    )
    ax1.plot(
        x[1:],
        exponential(x[1:], *popt),
        color=crm.plotting.COLOR1,
        label="ER",
    )
    ax1.fill_between(
        x[1:],
        exponential(x[1:], *[popt[i] - pcov[i][i] ** 0.5 for i in range(len(popt))]),
        exponential(x[1:], *[popt[i] + pcov[i][i] ** 0.5 for i in range(len(popt))]),
        color=crm.plotting.COLOR1,
        alpha=0.6,
    )
    ax1.set_xlabel("Time [min]")
    ax1.set_ylabel("Penalty [1/min]")
    ax2 = ax1.twinx()
    ax2.plot(x, n_cells, label="Cells", linestyle=(0, (1, 1)), color="k")
    ax2.set_ylabel("Number of Cells")
    ax2.set_ylim(1, 100)
    ax1.set_yscale("log")
    ax2.set_yscale("log")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2
    ax1.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=2,
        frameon=False,
    )

    path = Path("docs/source/_static/fitting-methods/")
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path / "progression-penalty-fitting.pdf"))
    print(f"{time.time() - interval:8.4} Plotted Results")
