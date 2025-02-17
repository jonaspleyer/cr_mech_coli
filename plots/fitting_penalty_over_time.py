import cr_mech_coli as crm
import matplotlib.pyplot as plt
from pathlib import Path
import time
import multiprocessing as mp
import scipy as sp
import numpy as np


def render_single_mask(n_iter: int, cell_container: str, domain_size, render_settings):
    cell_container = crm.CellContainer.deserialize(cell_container)
    cells_at_iter = cell_container.get_cells_at_iteration(n_iter)
    colors = cell_container.cell_to_color
    res = crm.render_mask(cells_at_iter, colors, domain_size, render_settings)
    return res


if __name__ == "__main__":
    config = crm.Configuration()
    config.t0 = 0.0
    config.dt = 0.1
    config.t_max = 200.0
    config.save_interval = 4.0
    config.n_agents = 4

    agent_settings = crm.AgentSettings(growth_rate=0.05)
    cell_container = crm.run_simulation(config, agent_settings)

    iterations = cell_container.get_all_iterations()

    interval = time.time()
    pool = mp.Pool()

    rs = crm.RenderSettings(resolution=800)
    args = [(i, cell_container.serialize(), config.domain_size, rs) for i in iterations]
    masks = pool.starmap(render_single_mask, args)
    print(f"{time.time() - interval:8.4} Calculated Masks:")
    interval = time.time()

    penalties_area_diff = [
        crm.penalty_area_diff(masks[i - 1], masks[i]) / config.save_interval
        for i in range(1, len(iterations))
    ]
    print(f"{time.time() - interval:8.4} Calculated Penalties without parents:")
    interval = time.time()

    penalties_parents = [
        crm.penalty_area_diff_account_parents(masks[i - 1], masks[i], cell_container, 0)
        / config.save_interval
        for i in range(1, len(iterations))
    ]
    print(f"{time.time() - interval:8.4} Calculated Penalties with parents:")
    interval = time.time()

    n_cells = [len(cell_container.get_cells_at_iteration(i)) for i in iterations]
    x = np.array([i * config.save_interval for i in range(len(iterations))])

    # Fit exponential function to penalties with parents
    def exponential(x, A, growth):
        return A * np.exp(growth * x)

    popt, pcov = sp.optimize.curve_fit(
        exponential, x[1:], penalties_parents, p0=(0.1, agent_settings.growth_rate)
    )

    fig, ax1 = plt.subplots()
    ax1.plot(x[1:], penalties_area_diff, label="Area Difference", linestyle=":", color="k")
    ax1.plot(x[1:], penalties_parents, label="Account for Parents", linestyle="-.", color="k")
    ax1.fill_between(
        x[1:],
        exponential(x[1:], *[popt[i] - pcov[i][i] ** 0.5 for i in range(len(popt))]),
        exponential(x[1:], *[popt[i] + pcov[i][i] ** 0.5 for i in range(len(popt))]),
        label="Fit $A e^{{\\lambda t}}$",
        color=(0.85, 0.85, 0.85),
    )
    ax1.legend(loc="upper left")
    ax1.set_xlabel("Time [min]")
    ax1.set_ylabel("Penalty [1/min]")
    ax2 = ax1.twinx()
    ax2.plot(x, n_cells, label="Number of Cells", linestyle=(0, (5, 7)), color="gray")
    ax2.legend(loc="upper right")
    ax2.set_ylabel("Number of Cells")
    fig.tight_layout()
    path = Path("docs/source/_static/fitting-methods/")
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path / "penalty-time-flow.png"))
    print(f"{time.time() - interval:8.4} Plotted Results")
    plt.show()
