import numpy as np
from glob import glob
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy as sp
import time
import argparse
import multiprocessing as mp

import cr_mech_coli as crm
from cr_mech_coli import crm_fit

data_dir = Path("data/crm_divide/0001/")

crm.plotting.set_mpl_rc_params()


def adjust_masks(
    masks, mask_iters, container: crm.CellContainer, settings, show_progress=False
):
    idents = container.cell_to_color.keys()
    iterations = container.get_all_iterations()
    idents_initial = container.get_cells_at_iteration(iterations[0]).keys()

    # 1. Identify new cells from simulation
    # 2. Link new cells with colors from later masks
    # 3. Extend cell_to_color and parent map

    # Map colors such that the cells which have not divided match with the previous masks
    # Mask color
    # 8 -> 5
    # 10 -> 6
    # In order to avoid collisions we also map divided cells to colors of value >= 20
    map_divided_cells_colors = {
        1: 20,
        2: 21,
        3: 22,
        4: 23,
        5: 24,
        6: 25,
        7: 26,
        9: 27,
        8: 5,
        10: 6,
    }
    # Since we know the explicit simulation snapshots,
    # we can write down the parent map directly
    parent_map_colors = {
        1: None,
        2: None,
        3: None,
        4: None,
        5: None,
        6: None,
        21: 1,
        23: 1,
        20: 2,
        22: 2,
        24: 3,
        26: 3,
        25: 4,
        27: 4,
    }

    def map_colors(m):
        for k, v in map_divided_cells_colors.items():
            m[m == k] = v
        return m

    x = np.array([len(np.unique(m)) - 1 for m in masks])
    n_divide = np.argmin(x < x[-1])

    masks = [m if n < n_divide else map_colors(m) for n, m in enumerate(masks)]

    # Map colors to Idents which are there initially
    # These idents do not change
    # mask_color_to_ident = {i: crm.CellIdentifier.new_initial(i) for i in range(1, 7)}
    # parent_map = {mask_color_to_ident[k]: None for k in range(1, 7)}

    mask_color_to_cell = {}

    # Calculate the average pixel for each color
    color_means = [
        {c: np.mean(np.where(m == c), axis=1) for c in np.unique(m)} for m in masks
    ]
    for id in idents:
        if id not in idents_initial:
            # Get history of cell
            cell_hist, parent_id = container.get_cell_history(id)
            # Obtain the first iteration at which the new cell appears
            first_iter = min(cell_hist.keys())
            # Obtain the very first position of this cell
            # and convert the position to pixel units
            first_pos = cell_hist[first_iter].pos
            first_pos = crm.convert_cell_pos_to_pixels(
                first_pos, settings.constants.domain_size, masks[0].shape
            )

            # Calculate the average of all vertices
            pos_mean = np.mean(first_pos, axis=0)[:2]

            # Obtain color for parent
            # This is identical to the CellIdentifier for initially constructed cells
            # with an offset of 1 since the color 0 is the background
            # Thus this code works (do not worry about type checking here)
            parent_color = parent_id[0] + 1

            # Since we have found the parent, we can see what colors its children have
            child_colors = list(
                filter(
                    lambda x: x is not None,
                    [
                        c if p == parent_color else None
                        for c, p in parent_map_colors.items()
                    ],
                )
            )

            # We obtain the index at which the first mask contains
            # the first ocurrance of the new CellIdentifier
            n_iter = np.argmax(first_iter < np.array(container.get_all_iterations()))
            n_ind = np.min(np.where(n_iter <= mask_iters))

            # We can now obtain the mean values of the children
            child_means = [color_means[n_ind][c] for c in child_colors]
            # Now we calculate the distance between the mean position of our agent
            # and the position of the candidates which is identical to the children
            # of the parent of the agent
            distances = [np.sum((pos_mean - cm) ** 2) ** 0.5 for cm in child_means]

            # We choose the child with the smaller distance as the correct one
            child_color = child_colors[np.argmin(distances)]
            mask_color_to_cell[child_color] = id

    cell_to_color = container.cell_to_color
    parent_map = container.parent_map

    # Now we extend the cell_to_color map and parent_map
    # Basically, we have to insert all cells that are not
    # results of the numerical simulation

    # We also adjust the masks such that the colors are
    # now matching.
    # For this, we must ensure that the colors which have
    # been assigned as noted in the CellContainer type
    # are correctly present.
    # This means, we use our various maps from above to
    # convert from mask colors to simulation colors

    color_data_transform = {np.uint8(0): (np.uint8(0), np.uint8(0), np.uint8(0))}

    process_later = []
    for color_flat in sorted(parent_map_colors.keys()):
        color_parent = parent_map_colors[color_flat]
        # Identify the cells which are not present in the simulation
        if (
            color_flat not in mask_color_to_cell.keys()
            and color_parent not in cell_to_color.keys()
            and color_parent is not None
        ):
            # By the condition above, we know that parent is not None
            # and thus the cell was newly created but is also not already
            # in the container. Thus we need to add the corresponding cell can obtain
            id = crm.CellIdentifier(crm.VoxelPlainIndex(0), color_flat)
            new_color = crm.counter_to_color(len(cell_to_color) + 1)
            cell_to_color[id] = new_color
            parent_map[id] = crm.CellIdentifier.new_initial(color_parent - 1)
            color_data_transform[color_flat] = new_color
        else:
            process_later.append(color_flat)

    for color_flat in process_later:
        color_data_transform[color_flat] = crm.counter_to_color(
            len(color_data_transform)
        )

    # Ensure that the parent_map does now contain all colors except for zero
    for c in [np.unique(m) for m in masks]:
        for ci in c:
            assert ci in parent_map_colors.keys() or ci == 0

    # Create new masks with updated colors
    new_masks = []
    if show_progress:
        iterator = tqdm(masks, total=len(masks), desc="Adjusting Data Masks")
    else:
        iterator = masks
    for m in iterator:
        new_mask = np.array(
            [color_data_transform[c] for c in m.reshape(-1)], dtype=np.uint8
        ).reshape((*m.shape, 3))
        new_masks.append(new_mask)

    color_to_cell = {v: k for k, v in cell_to_color.items()}

    return new_masks, parent_map, cell_to_color, color_to_cell


def predict(
    initial_positions,
    settings,
    radius=8.059267,
    strength=10.584545,
    bound=10,
    cutoff=100,
    en=0.50215733,
    em=0.21933548,
    diffusion_constant=0.0,
    spring_tension=3.0,
    rigidity=10.0,
    damping=2.5799131,
    growth_rates=[
        0.001152799,
        0.001410604,
        0.0018761827,
        0.0016834959,
        0.0036106023,
        0.0015209642,
    ],
    spring_length_thresholds: float | list[float] = [
        200.0,
        200.0,
        8.0,
        11.0,
        200.0,
        200.0,
    ],
    growth_rate_distrs=[
        (0.001152799, 0),
        (0.001410604, 0),
        (0.0018761827, 0),
        (0.0016834959, 0),
    ],
    show_progress=False,
):
    if np.any(np.array(spring_length_thresholds) <= radius):
        raise ValueError

    # Define agents
    interaction = crm.MiePotentialF32(
        radius,
        strength,
        bound,
        cutoff,
        en,
        em,
    )

    if type(spring_length_thresholds) is float:
        spring_length_thresholds = [spring_length_thresholds] * len(initial_positions)
    elif type(spring_length_thresholds) is list:
        pass
    else:
        raise TypeError("Expected float or list")

    if growth_rate_distrs is None:
        growth_rate_distrs = [(growth_rate, 0.0) for growth_rate in growth_rates[:4]]

    def spring_length(pos):
        dx = np.linalg.norm(pos[1:] - pos[:-1], axis=1)
        return np.mean(dx)

    agents = [
        crm.RodAgent(
            pos,
            vel=0 * pos,
            interaction=interaction,
            diffusion_constant=diffusion_constant,
            spring_tension=spring_tension,
            rigidity=rigidity,
            spring_length=spring_length(pos),
            damping=damping,
            growth_rate=growth_rate,
            growth_rate_distr=growth_rate_distr,
            spring_length_threshold=spring_length_threshold,
            neighbor_reduction=None,
        )
        for spring_length_threshold, pos, growth_rate, growth_rate_distr in zip(
            spring_length_thresholds,
            initial_positions,
            growth_rates,
            growth_rate_distrs,
        )
    ]

    # define config
    config = settings.to_config()
    config.show_progressbar = show_progress
    if show_progress:
        print()

    container = crm.run_simulation_with_agents(config, agents)
    return container


def objective_function(
    spring_length_thresholds_and_new_growth_rates,
    positions_initial,
    settings,
    masks_data,
    mask_iters,
    iterations_data,
    parent_penalty=1.0,
    return_all=False,
    return_times=False,
):
    times = [(time.perf_counter_ns(), "Start")]

    def update_time(message):
        if return_times:
            now = time.perf_counter_ns()
            times.append((now, message))

    spring_length_thresholds = spring_length_thresholds_and_new_growth_rates[:4]
    new_growth_rates = [
        *spring_length_thresholds_and_new_growth_rates[4:],
        # These should not come into effect at all
        0.0,
        0.0,
    ]

    try:
        container = predict(
            positions_initial,
            settings,
            spring_length_thresholds=[*spring_length_thresholds, 200.0, 200.0],
            growth_rate_distrs=[(g, 0) for g in new_growth_rates],
        )
    except ValueError as e:
        if return_all:
            raise e
        return np.inf
    iterations_simulation = np.array(container.get_all_iterations()).astype(int)

    update_time("Prediction")

    try:
        new_masks, parent_map, cell_to_color, color_to_cell = adjust_masks(
            masks_data, mask_iters, container, settings
        )
    except:
        return np.inf

    masks_predicted = [
        crm.render_mask(
            container.get_cells_at_iteration(iter),
            cell_to_color,
            settings.constants.domain_size,
            render_settings=crm.RenderSettings(pixel_per_micron=1),
        )
        for iter in iterations_simulation
    ]

    update_time("Masks")

    penalties = [
        crm.penalty_area_diff_account_parents(
            new_mask,
            masks_predicted[iter],
            color_to_cell,
            parent_map,
            parent_penalty,
        )
        for iter, new_mask in zip(iterations_data, new_masks)
    ]

    update_time("Penalties")

    if return_all:
        return (
            new_masks,
            parent_map,
            cell_to_color,
            color_to_cell,
            container,
            masks_predicted,
            penalties,
        )

    cost = np.sum(penalties)

    if return_times:
        return times

    return cost


def preprocessing():
    files_images = sorted(glob(str(data_dir / "images/*")))
    files_masks = sorted(glob(str(data_dir / "masks/*.csv")))
    masks = [np.loadtxt(fm, delimiter=",", dtype=np.uint8) for fm in files_masks]
    mask_iters = np.array([int(s[-10:-4]) for s in files_images])
    mask_iters = mask_iters - np.min(mask_iters)

    settings = crm_fit.Settings.from_toml(data_dir / "settings.toml")
    n_vertices = settings.constants.n_vertices

    iterations_all = []
    positions_all = []
    lengths_all = []
    colors_all = []
    for mask, filename in tqdm(
        zip(masks, files_masks), total=len(masks), desc="Extract positions"
    ):
        try:
            pos, length, _, colors = crm.extract_positions(
                mask, n_vertices, domain_size=settings.constants.domain_size
            )
            positions_all.append(np.array(pos, dtype=np.float32))
            lengths_all.append(length)
            iterations_all.append(int(Path(filename).stem.split("-")[0]))
            colors_all.append(colors)
        except ValueError as e:
            print("Encountered Error during extraction of positions:")
            print(filename)
            print(e)
            print("Omitting this particular result.")

    iterations_all = np.array(iterations_all, dtype=np.uint64) - iterations_all[0]
    settings.constants.n_saves = max(iterations_all)

    positions_initial = positions_all[0]
    domain_height = settings.domain_height
    positions_initial = np.append(
        positions_initial,
        domain_height / 2 + np.zeros((*positions_initial.shape[:2], 1)),
        axis=2,
    ).astype(np.float32)

    return masks, positions_initial, settings, iterations_all, mask_iters


def plot_time_evolution(
    masks_predicted,
    new_masks,
    color_to_cell,
    parent_map,
    iterations_simulation,
    iterations_data,
    settings,
    output_dir,
):
    fig, ax = plt.subplots(figsize=(8, 8))
    crm.plotting.configure_ax(ax)

    for color, parent_penalty in [
        (crm.plotting.COLOR1, 0),
        (crm.plotting.COLOR2, 0.5),
        (crm.plotting.COLOR3, 1.0),
    ]:
        penalties = [
            crm.penalty_area_diff_account_parents(
                new_mask,
                masks_predicted[iter],
                color_to_cell,
                parent_map,
                parent_penalty,
            )
            for iter, new_mask in tqdm(
                zip(iterations_data, new_masks),
                total=len(new_masks),
                desc="Calculating penalties",
            )
        ]
        ax.plot(
            np.array([iterations_simulation[i] for i in iterations_data])
            * settings.constants.dt,
            penalties,
            marker="x",
            color=color,
            label=f"p={parent_penalty}",
        )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=3,
        frameon=False,
    )
    ax.set_ylabel("Cost Function")
    ax.set_xlabel("Time [h]")
    fig.savefig(output_dir / "crm_divide.pdf")
    fig.savefig(output_dir / "crm_divide.png")
    plt.close(fig)


def plot_timings(
    parameters,
    positions_initial,
    settings,
    masks_data,
    mask_iters,
    iterations_data,
    output_dir,
    n_samples: int = 3,
):
    times = []
    for _ in tqdm(range(n_samples), total=n_samples, desc="Measure Timings"):
        times.append(
            # [("p0", 1 * n), ("p1", 2 * n), ("p2", 3 * n)]
            objective_function(
                parameters,
                positions_initial,
                settings,
                masks_data,
                mask_iters,
                iterations_data,
                parent_penalty=1.0,
                return_times=True,
            )
        )

    data = np.array(
        [[times[i][j][0] for j in range(len(times[0]))] for i in range(len(times))]
    )
    data = (data[:, 1:] - data[:, :-1]) / 1e6
    mean = np.mean(data, axis=0)
    labels = [t[1] for t in times[0][1:]]

    fig, ax = plt.subplots(figsize=(8, 8))
    crm.configure_ax(ax)
    ax.bar(labels, mean, color=crm.plotting.COLOR3)
    ax.set_yscale("log")
    ax.set_ylabel("Time [ms]")
    fig.savefig(output_dir / "timings.pdf")
    fig.savefig(output_dir / "timings.png")


def main():
    parser = argparse.ArgumentParser(
        description="Fits the Bacterial Rods model to a system of cells."
    )
    parser.add_argument(
        "-i",
        "--iteration",
        type=int,
        default=None,
        help="Use existing output folder instead of creating new one",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="out/crm_divide/",
        help="Directory where to store results",
    )
    parser.add_argument(
        "--skip-time-evolution",
        action="store_true",
        help="Skip plotting of the time evolution of costs",
    )
    parser.add_argument(
        "--skip-timings",
        action="store_true",
        help="Skip plotting of the timings",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=-1,
        help="Number of threads to utilize",
    )
    pyargs = parser.parse_args()

    n_workers = pyargs.workers
    if n_workers <= 0:
        n_workers = mp.cpu_count()

    iteration = pyargs.iteration
    if pyargs.iteration is None:
        existing = glob(f"{pyargs.output_dir}/*")
        if len(existing) == 0:
            iteration = 0
        else:
            iteration = max([int(Path(i).name) for i in existing]) + 1
    output_dir = Path(f"{pyargs.output_dir}/{iteration:06}")

    # Create the directory if we had to choose a new one
    if pyargs.iteration is None:
        output_dir.mkdir(parents=True)

    masks_data, positions_initial, settings, iterations_data, mask_iters = (
        preprocessing()
    )

    spring_length_thresholds = [12.0] * 4
    new_growth_rates = [
        0.001152799,
        0.001410604,
        0.0018761827,
        0.0016834959,
    ]
    spring_length_thresholds_and_new_growth_rates = [
        *spring_length_thresholds,
        *new_growth_rates,
    ]
    bounds = [(5.0, 30.0)] * 4 + [(0.001, 0.002)] * 4
    parent_penalty = 0.5
    args = (
        positions_initial,
        settings,
        masks_data,
        mask_iters,
        iterations_data,
        parent_penalty,
    )

    # Try loading data
    if pyargs.iteration is not None:
        result = np.loadtxt(output_dir / "optimize_result.csv")
        final_parameters = result[:-1]
        final_cost = result[-1]
    else:
        res = sp.optimize.differential_evolution(
            objective_function,
            x0=spring_length_thresholds_and_new_growth_rates,
            bounds=bounds,
            args=args,
            disp=True,
            maxiter=4,
            popsize=20,
            mutation=(0.6, 1),
            recombination=0.5,
            workers=n_workers,
            updating="deferred",
            polish=True,
        )
        final_parameters = res.x
        final_cost = res.fun
        np.savetxt(output_dir / "optimize_result.csv", [*final_parameters, final_cost])
    (
        new_masks,
        parent_map,
        cell_to_color,
        color_to_cell,
        container,
        masks_predicted,
        penalties,
    ) = objective_function(final_parameters, *args, return_all=True)

    plot_time_evolution(
        masks_predicted,
        new_masks,
        color_to_cell,
        parent_map,
        container.get_all_iterations(),
        iterations_data,
        settings,
        output_dir,
    )

    if not pyargs.skip_timings:
        plot_timings(
            final_parameters,
            positions_initial,
            settings,
            masks_data,
            mask_iters,
            iterations_data,
            output_dir,
        )
