import cr_mech_coli as crm
import matplotlib.pyplot as plt
from pathlib import Path
import time
import multiprocessing as mp
import scipy as sp
import numpy as np
import argparse

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
        crm.penalty_area_diff_account_parents(masks[i - 1], masks[i], cell_container, 0)
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
    fig, ax1 = plt.subplots(figsize=(8, 8))
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
        bbox_to_anchor=(0.5, 1.18),
        ncol=2,
        frameon=False,
    )

    path = Path("docs/source/_static/fitting-methods/")
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path / "penalty-time-flow.png"))
    fig.savefig(str(path / "penalty-time-flow.pdf"))
    print(f"{time.time() - interval:8.4} Plotted Results")
