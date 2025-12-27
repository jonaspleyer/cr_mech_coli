import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

import cr_mech_coli as crm
from cr_mech_coli import COLOR1, COLOR2, COLOR3, COLOR4, COLOR5


def render_image(
    iteration: int,
    render_settings,
    cell_container_serialized: list[int],
    domain_size,
    out_path: Path,
):
    container = crm.CellContainer.deserialize(cell_container_serialized)
    cells = container.get_cells_at_iteration(iteration)
    colors = {
        key: [
            0,
            min(
                255,
                int(
                    np.round(
                        255 * np.max(value[0].pos[:, 2]) / (value[0].radius * 2 * 2)
                    )
                ),
            ),
            0,
        ]
        for (key, value) in cells.items()
    }
    crm.render_pv_image(
        cells,
        render_settings,
        domain_size,
        colors,
        filename=out_path / f"{iteration:010}.png",
    )


def render_image_helper(args):
    render_image(*args)


def plot_colony_height_over_time():
    ml_config = produce_ml_config()

    def create_new_ml_configs(ml_config, seeds):
        for seed in seeds:
            ml_config_new = ml_config.clone_with_args(rng_seed=seed)
            ml_config_new.config.storage_suffix = f"{seed:03}"
            yield ml_config_new

    # Produce data for various configs
    ml_configs = list(create_new_ml_configs(ml_config, pyargs.seeds))

    iterations = []
    ymax_values = []
    y95th_values = []
    ymean_values = []
    n_agents = []
    for ml_config in ml_configs:
        container = load_or_compute(ml_config)
        out_path = container.path if container.path is not None else exit()

        i, positions, ymax, y95th, ymean = produce_ydata(container)
        n_agents.append([p.shape[0] for p in positions])
        iterations.append(i)
        ymax_values.append(ymax)
        y95th_values.append(y95th)
        ymean_values.append(ymean)

        if pyargs.plot_snapshots:
            # Define a maximum resolution of 800 pixels
            ppm = 1200 / np.max(ml_config.config.domain_size)
            render_settings = crm.RenderSettings(pixel_per_micron=ppm)
            cell_container_serialized = container.serialize()
            pool = mp.Pool()
            args = [
                (
                    i,
                    render_settings,
                    cell_container_serialized,
                    ml_config.config.domain_size,
                    out_path,
                )
                for i in container.get_all_iterations()
            ]

            _ = list(
                tqdm(
                    pool.imap(render_image_helper, args),
                    total=len(args),
                    desc=str(out_path.stem),
                )
            )

    fig, ax = plt.subplots(figsize=(8, 8))

    t = np.array(iterations[0]) * ml_config.config.dt
    ymax = np.mean(ymax_values, axis=0)
    ymax_err = np.std(ymax_values, axis=0)
    y95th_std = np.mean(y95th_values, axis=0)
    y95th_err = np.std(y95th_values, axis=0)
    ymean_std = np.mean(ymean_values, axis=0)
    ymean_err = np.std(ymean_values, axis=0)
    n_agents = np.array(n_agents)
    diameter = 2 * ml_config.agent_settings.interaction.radius

    ax.plot(t, ymax, label="Max", c=COLOR3)
    ax.fill_between(t, ymax - ymax_err, ymax + ymax_err, color=COLOR3, alpha=0.3)

    ax.plot(t, y95th_std, label="95th pctl.", c=COLOR1)
    ax.fill_between(
        t, y95th_std - y95th_err, y95th_std + y95th_err, color=COLOR1, alpha=0.3
    )

    ax.plot(t, ymean_std, label="Mean", c=COLOR5)
    ax.fill_between(
        t, ymean_std - ymean_err, ymean_std + ymean_err, color=COLOR5, alpha=0.3
    )

    yticks = diameter * np.arange(np.ceil(np.max(ymax) / diameter))
    yticklabels = [i + 1 for i, _ in enumerate(yticks)]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.grid(True, which="major", linestyle="-", linewidth=0.75, alpha=0.75)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle="-", linewidth=0.25, alpha=0.15)

    ax.set_ylabel("Colony Height [Cell Diameter]")
    ax.set_xlabel("Time [min]")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=3,
        frameon=False,
    )

    fig.savefig("out/crm_multilayer/multilayer-time-evolution.pdf")
    fig.savefig("out/crm_multilayer/multilayer-time-evolution.png")
