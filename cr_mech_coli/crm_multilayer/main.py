"""
TODO
"""

from matplotlib.colors import hex2color
import cr_mech_coli as crm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import multiprocessing as mp
import argparse
from tqdm import tqdm
from pathlib import Path
from glob import glob

from cr_mech_coli.crm_multilayer import MultilayerConfig
from cr_mech_coli.crm_perf_plots import COLOR1, COLOR2, COLOR3, COLOR4, COLOR5


def run_sim(ml_config: MultilayerConfig) -> crm.CellContainer:
    positions = np.array(
        crm.generate_positions(
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


def produce_ydata(container: crm.CellContainer):
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
    settings_files2 = glob(str(out_path / "*/*/ml_config.toml"))
    settings_files.extend(settings_files2)

    for file_path in settings_files:
        file_path = Path(file_path)
        ml_config_loaded = MultilayerConfig.load_from_toml_file(Path(file_path))
        if ml_config.approx_eq(ml_config_loaded):
            container = crm.CellContainer.load_from_storage(
                ml_config.config, file_path.parent
            )
            return container
    else:
        res = run_sim(ml_config)
        print()
        return res


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


def produce_ml_config() -> MultilayerConfig:
    # Create many Multilayer-Configs
    ml_config = crm.crm_multilayer.MultilayerConfig()
    ml_config.config.dt = 0.05
    ml_config.config.t_max = 350
    ml_config.config.n_saves = int(
        np.ceil(ml_config.config.t_max / (ml_config.config.dt * 100))
    )
    ml_config.config.domain_height = 20.0
    ml_config.config.domain_size = (1600, 1600)
    ml_config.dx = (700, 700)
    ml_config.config.n_voxels = (10, 10)
    ml_config.config.gel_pressure = 0.05
    ml_config.config.n_threads = 1

    ml_config.config.surface_friction = 0.3
    ml_config.config.surface_friction_distance = (
        ml_config.agent_settings.interaction.radius / 10
    )

    ml_config.agent_settings.mechanics.damping = 0.1
    ml_config.agent_settings.mechanics.rigidity = 15
    ml_config.agent_settings.interaction.strength = 0.2
    ml_config.agent_settings.neighbor_reduction = (200, 0.5)
    ml_config.agent_settings.growth_rate = 0.4
    ml_config.agent_settings.growth_rate_distr = (0.4, 0.02)

    ml_config.config.storage_options = [
        crm.simulation.StorageOption.Memory,
        crm.simulation.StorageOption.SerdeJson,
    ]
    ml_config.config.storage_location = "out/crm_multilayer"

    return ml_config


def plot_colony_height_over_time():
    parser = argparse.ArgumentParser(
        prog="crm_multilayer",
        description="Run Simulations to analyze Multilayer-behaviour of Rod-Shaped Bacteria.",
    )
    parser.add_argument("--plot-snapshots", action="store_true")
    parser.add_argument("--seeds", nargs="+", default=[0, 1, 2, 3], type=int)
    pyargs = parser.parse_args()
    pyargs.seeds = [int(n) for n in pyargs.seeds]

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


def produce_ydata_helper(ml_config_string):
    ml_config = MultilayerConfig.load_from_toml_str(ml_config_string)
    container = load_or_compute(ml_config)
    return produce_ydata(container)


def plot_colony_height_versus_gel_pressure():
    ml_config = produce_ml_config()
    ml_config.config.n_saves = 100
    ml_config.config.dt *= 1.5
    ml_config.config.t_max = 200

    gel_pressures = np.arange(0.1, 0.525, 0.025)
    seeds = np.arange(8)

    def create_args(ml_config, gel_pressures, seeds):
        for gel_pressure in gel_pressures:
            for s in seeds:
                ml_config_new = ml_config.clone_with_args()
                ml_config_new.config.gel_pressure = gel_pressure
                ml_config_new.config.rng_seed = s
                ml_config_new.config.storage_suffix = (
                    f"seed{s:02}-strength{gel_pressure:08.5f}"
                )
                yield ml_config_new.to_toml_string()

    args = list(create_args(ml_config, gel_pressures, seeds))

    n_threads = mp.cpu_count() // ml_config.config.n_threads
    pool = mp.Pool(n_threads)
    data = list(tqdm(pool.imap(produce_ydata_helper, args), total=len(args)))

    data_times = (
        np.array([d[0] for d in data]).reshape((len(gel_pressures), len(seeds), -1))
        * ml_config.config.dt
    )
    data_ymax = np.array([d[2] for d in data]).reshape(data_times.shape)
    data_y95th = np.array([d[3] for d in data]).reshape(data_times.shape)

    radius = ml_config.agent_settings.interaction.radius

    ind_max = np.argmin(data_ymax < 1.5 * radius, axis=2)
    ind_y95th = np.argmin(data_y95th < 1.5 * radius, axis=2)

    times_max = np.zeros(ind_max.shape)
    times_95th = np.zeros(ind_max.shape)

    for i in range(data_ymax.shape[0]):
        for j in range(data_ymax.shape[1]):
            times_max[i, j] = data_times[i][j][ind_max[i, j]]
            times_95th[i, j] = data_times[i][j][ind_y95th[i, j]]

    ############
    ## Plot 1 ##
    ############
    fig, ax = plt.subplots(figsize=(8, 8))

    times = [times_max, times_95th]
    colors = [COLOR3, COLOR5]
    labels = ["Max", "95th pctl."]
    for t, color, label in zip(times, colors, labels):
        t_mean = np.mean(t, axis=1)
        t_err = np.std(t, axis=1)

        ax.plot(gel_pressures, t_mean, c=color, linestyle="-", label=label)
        ax.fill_between(
            gel_pressures,
            t_mean - t_err,
            t_mean + t_err,
            color=color,
            alpha=0.3,
        )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=3,
        frameon=False,
    )

    ############
    ## Plot 2 ##
    ############
    fig2, ax2 = plt.subplots(figsize=(8, 8))

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "mymap", [(0.0, hex2color(COLOR1)), (1.0, hex2color(COLOR3))]
    )

    y_collection = [
        np.column_stack([np.mean(t, axis=0), np.mean(d, axis=0) / radius])
        for t, d in zip(data_times, data_y95th)
    ]

    line_collection = mpl.collections.LineCollection(
        y_collection, array=gel_pressures, cmap=cmap
    )

    ax2.set_xlim(float(np.min(data_times)), float(np.max(data_times)))
    ylow = float(np.min([y[:, 1] for y in y_collection]))
    yhigh = float(np.max([y[:, 1] for y in y_collection]))
    diff = yhigh - ylow
    ax2.set_ylim(ylow - 0.05 * diff, yhigh + 0.05 * diff)

    ax2.add_collection(line_collection)
    fig2.colorbar(line_collection, label="Gel Pressure")

    for a in [ax, ax2]:
        a.grid(True, which="major", linestyle="-", linewidth=0.75, alpha=0.75)
        a.minorticks_on()
        a.grid(True, which="minor", linestyle="-", linewidth=0.25, alpha=0.15)

    # Save Plot 1
    ax.set_xlabel("Gel Pressure")
    ax.set_ylabel("Transition 2nd Layer [min]")
    fig.savefig("out/crm_multilayer/colony-height-vs-gel_pressure.pdf")
    fig.savefig("out/crm_multilayer/colony-height-vs-gel_pressure.png")

    # Save Plot 2
    ax2.set_xlabel("Time [min]")
    ax2.set_ylabel("95th pctl. Colony Height [R]")
    fig2.savefig("out/crm_multilayer/colony-height-vs-time.pdf")
    fig2.savefig("out/crm_multilayer/colony-height-vs-time.png")


def crm_multilayer_main():
    set_rc_params()
    plot_colony_height_over_time()
    plot_colony_height_versus_gel_pressure()
