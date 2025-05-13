"""
TODO
"""

import cr_mech_coli as crm
from cr_mech_coli.plotting import COLOR1, COLOR2, COLOR3, COLOR4, COLOR5
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import scipy as sp


def run_sim(
    n_agents: int,
    domain_size: float,
    rng_seed: int = 0,
    n_threads: int = 1,
    n_voxels: int = 1,
):
    config = crm.Configuration(
        domain_size=(domain_size + 100, domain_size + 100),
        n_threads=n_threads,
        n_voxels=(n_voxels, n_voxels),
    )
    agent_settings = crm.AgentSettings()
    positions = crm.generate_positions(
        n_agents,
        agent_settings,
        config,
        rng_seed,
        dx=(50, 50),
        randomize_positions=0.05,
    )
    agents = [
        crm.RodAgent(pos=p, vel=0 * p, **agent_settings.to_rod_agent_dict())
        for p in positions
    ]

    t_start = time.time()
    container = crm.run_simulation_with_agents(config, agents)
    last_iter = container.get_all_iterations()[-1]
    return time.time() - t_start, len(container.get_cells_at_iteration(last_iter))


def crm_perf_plots_main():
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

    n_agents = np.array([2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64])
    n_seeds = np.arange(0, 3)
    n_threads = np.array([1, 2, 4])
    args = [(n, s, nt) for n in n_agents for s in n_seeds for nt in n_threads]
    domain_size = 300

    n_agents_fin = []
    y = []
    for arg in tqdm(args):
        (n, s, nt) = arg
        t, n_fin = run_sim(n, domain_size, rng_seed=s, n_threads=nt, n_voxels=6)
        y.append(t)
        n_agents_fin.append(n_fin)

    y = np.array(y).reshape(len(n_agents), len(n_seeds), len(n_threads))
    n_agents = np.mean(
        np.array(n_agents_fin).reshape(len(n_agents), len(n_seeds), len(n_threads)),
        axis=1,
    )

    data = np.mean(y, axis=1)
    err = np.std(y, axis=1)

    linestyles1 = [
        (0, (10, 0)),
        (0, (4, 5)),
        (0, (2, 3)),
    ]
    linestyles2 = [
        (0, (10, 0)),
        (4.5, (4, 5)),
        (2.5, (2, 3)),
    ]
    fig, ax = plt.subplots(figsize=(8, 8))

    for i, n_threads in enumerate(n_threads):
        ax.plot(
            n_agents[:, i],
            data[:, i],
            label=f"{n_threads}T",
            color=COLOR3,
            linestyle=linestyles1[i],
        )
        ax.fill_between(
            n_agents[:, i],
            data[:, i] - err[:, i],
            data[:, i] + err[:, i],
            color=COLOR1,
            alpha=0.3,
        )

        popt, _ = sp.optimize.curve_fit(
            lambda t, a, b: a * t**2 + b,
            n_agents[:, i],
            data[:, i],
            absolute_sigma=True,
        )
        (a, b) = popt
        y_fit = a * n_agents[:, i] ** 2 + b
        ax.plot(
            n_agents[:, i], y_fit, label="QR", color=COLOR5, linestyle=linestyles2[i]
        )

    ax.set_xlim(np.min(n_agents).astype(float), np.max(n_agents).astype(float))
    ax.set_ylabel("Wall Time [s]")
    ax.set_xlabel("Number of Final Agents")

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=3,
        frameon=False,
    )

    ax.grid(True, which="major", linestyle="-", linewidth=0.75, alpha=0.25)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle="-", linewidth=0.25, alpha=0.15)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig("computation-time-with-initial-agents.pdf")
    fig.savefig("computation-time-with-initial-agents.png")
