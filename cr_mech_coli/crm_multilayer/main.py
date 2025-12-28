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

from cr_mech_coli.crm_multilayer import MultilayerConfig
from cr_mech_coli import COLOR1, COLOR2, COLOR3, COLOR4, COLOR5

from .runner import load_or_compute_container, produce_ml_config, produce_ydata


def produce_ydata_helper(ml_config_string):
    ml_config = MultilayerConfig.load_from_toml_str(ml_config_string)
    container = load_or_compute_container(ml_config)
    return produce_ydata(container)


def plot_colony_height_versus_gel_pressure():
    ml_config = produce_ml_config()
    ml_config.config.n_saves = 100
    ml_config.config.dt *= 1.5
    ml_config.config.t_max = 20

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


def plot_elevation_map(ml_config, cells: dict, dx):
    nx = int(ml_config.config.domain_size[0] / dx)
    ny = int(ml_config.config.domain_size[1] / dx)
    zmax = np.zeros((nx, ny))

    positions = np.array([c[0].pos for c in cells.values()])

    for pi in positions:
        for pii in pi:
            ind_x = int(pii[0] / ml_config.config.domain_size[0] * nx)
            ind_y = int(pii[1] / ml_config.config.domain_size[1] * ny)
            zmax[ind_x, ind_y] = max(zmax[ind_x, ind_y], pii[2])

    mask = zmax == 0
    zmax /= dx
    zmax[~mask] += 1

    fig, ax = plt.subplots(figsize=(nx, ny))
    ax.imshow(
        zmax.T[::-1],
        vmin=0,
        extent=(0, ml_config.config.domain_size[0], 0, ml_config.config.domain_size[1]),
        # interpolation="nearest",
    )  # , vmax=(int(np.max(zmax)) + 1))

    positions = np.array([c[0].pos for c in cells.values()])
    dirs = positions[:, 1:] - positions[:, 0:-1]
    middle = (positions[:, 1:] + positions[:, 0:-1]) / 2

    # fig, ax = plt.subplots(figsize=(8, 8))
    for i in range(positions.shape[0]):
        ax.plot(positions[i, :, 0], positions[i, :, 1], color="k")

    ax.set_xlim(0, ml_config.config.domain_size[0])
    ax.set_ylim(0, ml_config.config.domain_size[1])

    plt.show()


def plot_colony_height(ml_config: MultilayerConfig, container):
    iterations = container.get_all_iterations()

    zmax = []
    n_cells = []
    for i in iterations:
        cells = container.get_cells_at_iteration(i)
        positions = np.array([c[0].pos for c in cells.values()])
        zmax.append(np.max(np.mean(positions[:, :, 2], axis=1)))
        n_cells.append(positions.shape[0])

    t = ml_config.config.t0 + np.array(iterations) * ml_config.config.dt / 60
    y1 = np.array(zmax) / ml_config.agent_settings.interaction.radius / 2
    y2 = np.array(n_cells)

    fig, ax = plt.subplots(figsize=(8, 8))
    crm.configure_ax(ax, minor=False)
    ax.plot(t, y1, color=crm.COLOR3, label="Colony Height")
    ax2 = ax.twinx()
    crm.configure_ax(ax2, minor=False)
    ax2.plot(t, y2, color=crm.COLOR5, label="Number of Cells")

    labels1, handles1 = ax.get_legend_handles_labels()
    labels2, handles2 = ax2.get_legend_handles_labels()

    ax.set_yticks([0, 1])
    ax.legend(
        [*labels1, *labels2],
        [*handles1, *handles2],
        ncol=2,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
    )

    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Diameter [2R]")
    ax2.set_ylabel("Number of Cells")

    plt.show()


def crm_multilayer_main():
    parser = argparse.ArgumentParser(
        prog="crm_multilayer",
        description="Run Simulations to analyze Multilayer-behaviour of Rod-Shaped Bacteria.",
    )
    subparsers = parser.add_subparsers()
    parse_run = subparsers.add_parser(
        "run", help="Run simulation for specified parameters"
    )
    parse_plot = subparsers.add_parser("plot", help="Perform plotting actions")

    parse_plot.add_argument("colony-height", action="store_true", default=False)
    # parse_plot.add_argument("", action="store_true", default=False)

    # parser.add_argument("--plot-snapshots", action="store_true")
    # parser.add_argument("--seeds", nargs="+", default=[0, 1, 2, 3], type=int)
    pyargs = parser.parse_args()
    # pyargs.seeds = [int(n) for n in pyargs.seeds]

    # ml_config = MultilayerConfig.load_from_toml_file(Path(file_path))
    ml_config = produce_ml_config()
    ml_config.config.progressbar = "Run Simulation"

    container = load_or_compute_container(ml_config)

    # for i in tqdm(container.get_all_iterations()[-2:]):
    #     render_image(
    #         i,
    #         RenderSettings(pixel_per_micron=12),
    #         container.serialize(),
    #         ml_config.config.domain_size,
    #         Path("images"),
    #     )

    # plot_elevation_map(
    #     ml_config,
    #     container.get_cells_at_iteration(container.get_all_iterations()[-1]),
    #     dx=2 * ml_config.agent_settings.interaction.radius,
    # )

    # plot_vector_field(
    #     ml_config,
    #     container.get_cells_at_iteration(container.get_all_iterations()[-1]),
    # )

    crm.set_mpl_rc_params()
    plot_colony_height(ml_config, container)

    exit()

    # plot_colony_height_over_time()
    # plot_colony_height_versus_gel_pressure()
