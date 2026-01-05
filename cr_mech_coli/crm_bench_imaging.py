import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

import cr_mech_coli as crm
from .cr_mech_coli import render_mask_vtk


def get_timings(seed: int = 0, ppm=10, n_saves=6):
    config = crm.Configuration()
    agent_settings = crm.AgentSettings(
        growth_rate=0.078947365,
    )
    config.t0 = 0.0
    config.dt = 0.1
    config.t_max = 200.0
    config.n_saves = n_saves - 2
    config.domain_size = (np.float32(150.0), np.float32(150.0))

    positions = crm.generate_positions(
        n_agents=4,
        agent_settings=agent_settings,
        config=config,
        rng_seed=seed,
        dx=(config.domain_size[0] * 0.1, config.domain_size[1] * 0.1),
    )
    agent_dict = agent_settings.to_rod_agent_dict()
    cells = [crm.RodAgent(p, 0.0 * p, **agent_dict) for p in positions]

    cell_container = crm.run_simulation_with_agents(config, cells)

    render_settings = crm.RenderSettings(pixel_per_micron=ppm)

    resolution = (
        int(config.domain_size[0] * ppm),
        int(config.domain_size[1] * ppm),
    )

    times = []
    iterations = cell_container.get_all_iterations()
    for i in tqdm(iterations, desc="Gather Timings"):
        cells = cell_container.get_cells_at_iteration(i)
        start = time.time()
        m0 = crm.render_mask(
            cells,
            cell_container.cell_to_color,
            (config.domain_size[0], config.domain_size[1]),
            render_settings,
        )
        t1 = time.time() - start
        start = time.time()
        m1 = render_mask_vtk(
            cells,
            cell_container.cell_to_color,
            (config.domain_size[0], config.domain_size[1]),
            ppm,
        )
        t2 = time.time() - start
        start = time.time()
        m2, _ = crm.render_mask_2d(
            cells,
            cell_container.cell_to_color,
            (config.domain_size[0], config.domain_size[1]),
            resolution,
            delta_angle=np.float32(np.pi / 15.0),
        )
        t3 = time.time() - start
        times.append((len(cells), t1, t2, t3))

        assert m0.shape == m1.shape
        assert m1.shape == m2.shape

    return np.array(times)


def crm_bench_imaging():
    n_samples = 7
    n_saves = 10
    times_shape = (n_samples, n_saves, 4)
    try:
        times = np.load("timings.npy")
        assert times.shape == times_shape
    except:
        times = np.zeros(times_shape)
        for i1, s in enumerate(range(n_samples)):
            times[i1, :] = get_timings(s, n_saves=n_saves)
        times = np.array(times)
        np.save("timings", times)
    n_cells = times[:, :, 0]

    crm.set_mpl_rc_params()
    fig, ax = plt.subplots(figsize=(8, 8))
    crm.configure_ax(ax)

    # Plot against time of simulation
    colors = [crm.COLOR3, crm.COLOR1, crm.COLOR5]
    for i, color, label in zip(range(1, 4), colors, ["pyvista", "vtk", "plotters"]):
        mean = np.mean(times[:, :, i], axis=0)
        std = np.std(times[:, :, i], axis=0)
        ax.plot(range(len(mean)), mean, label=label, color=color)
        ax.fill_between(
            range(len(mean)), mean - std, mean + std, color=color, alpha=0.5
        )
    ax2 = ax.twinx()
    ax2.plot(
        np.mean(n_cells, axis=0),
        color="black",
        label="Cells",
        linestyle="--",
    )
    ax2.set_ylabel("Number of Cells")

    ax.set_ylabel("Render Time [s]")
    ax.set_xlabel("Iteration")

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        labels=[*labels1, *labels2],
        handles=[*handles1, *handles2],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=2,
        frameon=False,
    )
    plt.show()
