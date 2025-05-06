import cr_mech_coli as crm
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import argparse
from tqdm import tqdm
from pathlib import Path
from glob import glob

from cr_mech_coli.crm_multilayer import MultilayerConfig
from cr_mech_coli.crm_perf_plots import COLOR1, COLOR2, COLOR3, COLOR4, COLOR5


def run_sim(ml_config: MultilayerConfig) -> crm.CellContainer:
    positions = np.array(
        crm.generate_positions_old(
            n_agents=1,
            agent_settings=ml_config.agent_settings,
            config=ml_config.config,
            rng_seed=ml_config.rng_seed,
            dx=ml_config.dx,
            randomize_positions=ml_config.randomize_positions,
            n_vertices=ml_config.n_vertices,
        )
    )
    positions[:, :, 2] = 0.1 * ml_config.agent_settings.interaction.radius
    agent_dict = ml_config.agent_settings.to_rod_agent_dict()

    agents = [crm.RodAgent(p, 0.0 * p, **agent_dict) for p in positions]

    container = crm.run_simulation_with_agents(ml_config.config, agents)
    if container.path is not None:
        ml_config.to_toml_file(Path(container.path) / "ml_config.toml")
    else:
        print("Could not find save path for MultilayerConfig:")
        print(ml_config.to_toml_string())
    return container


def produce_ydata(container):
    cells = container.get_cells()
    iterations = container.get_all_iterations()
    positions = [np.array([c[0].pos for c in cells[i].values()]) for i in iterations]
    ymax = np.array([np.max(p[:, :, 2]) for p in positions])
    y95th = np.array([np.percentile(p[:, :, 2], 95) for p in positions])
    ymean = np.array([np.mean(p[:, :, 2]) for p in positions])
    return iterations, positions, ymax, y95th, ymean


def load_or_compute(
    ml_config: MultilayerConfig, out_path=Path("out/crm_multilayer/")
) -> crm.CellContainer:
    settings_files = glob(str(out_path / "*/ml_config.toml"))
    for file_path in settings_files:
        file_path = Path(file_path)
        ml_config_loaded = MultilayerConfig.load_from_toml_file(Path(file_path))
        if ml_config.approx_eq(ml_config_loaded):
            container = crm.CellContainer.load_from_storage(
                ml_config.config, file_path.parent.stem
            )
            return container
    else:
        return run_sim(ml_config)


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


def set_rc_params():
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


def crm_multilayer_main():
    """
    TODO
    """
    parser = argparse.ArgumentParser(
        prog="crm_multilayer",
        description="Run Simulations to analyze Multilayer-behaviour of Rod-Shaped Bacteria.",
    )
    parser.add_argument("--plot-snapshots", default=False, type=bool)
    parser.add_argument("--seeds", nargs="+", default=[0, 1, 2], type=int)
    pyargs = parser.parse_args()
    pyargs.seeds = [int(n) for n in pyargs.seeds]

    # Create many Multilayer-Configs
    ml_config = crm.crm_multilayer.MultilayerConfig()
    ml_config.config.dt = 0.05
    ml_config.config.t_max = 250
    ml_config.config.n_saves = int(
        np.ceil(ml_config.config.t_max / (ml_config.config.dt * 100))
    )
    ml_config.config.domain_height = 20.0
    ml_config.config.domain_size = (1600, 1600)
    ml_config.dx = (700, 700)
    ml_config.config.n_voxels = (10, 10)
    ml_config.config.gravity = 0.15
    ml_config.config.show_progressbar = True
    ml_config.config.n_threads = 8

    ml_config.config.surface_friction = 0.3
    ml_config.config.surface_friction_distance = (
        ml_config.agent_settings.interaction.radius / 10
    )

    ml_config.agent_settings.mechanics.damping = 0.05
    ml_config.agent_settings.mechanics.diffusion_constant
    ml_config.agent_settings.mechanics.rigidity = 15
    ml_config.agent_settings.interaction.strength = 0.2
    ml_config.agent_settings.neighbor_reduction = (200, 0.5)
    ml_config.agent_settings.growth_rate = 0.4

    ml_config.config.storage_options = [
        crm.simulation.StorageOption.Memory,
        crm.simulation.StorageOption.SerdeJson,
    ]
    ml_config.config.storage_location = "out/crm_multilayer"

    ml_configs = [ml_config.clone_with_args(rng_seed=seed) for seed in pyargs.seeds]
    # Produce data for various configs

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

    set_rc_params()
    fig, ax = plt.subplots(figsize=(8, 8))

    t = np.array(iterations[0]) * ml_config.config.dt
    ymax = np.mean(ymax_values, axis=0)
    ymax_err = np.std(ymax_values, axis=0)
    y95th_std = np.mean(y95th_values, axis=0)
    y95th_err = np.std(y95th_values, axis=0)
    ymean_std = np.mean(ymean_values, axis=0)
    ymean_err = np.std(ymean_values, axis=0)
    n_agents = np.array(n_agents)
    radius = ml_config.agent_settings.interaction.radius * np.ones(t.shape[0])
    diameter = 2 * radius[0]

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

    fig.savefig("out/crm_multilayer/tmp.pdf")
