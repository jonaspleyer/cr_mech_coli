"""
.. code-block:: text
    :caption: Usage of the `crm_divide` script

    crm_divide -h

    usage: crm_divide [-h] [-i ITERATION] [--output-dir OUTPUT_DIR] [--skip-profiles] [--skip-time-evolution]
                    [--skip-snapshots] [--skip-timings] [--skip-mask-adjustment] [--only-mask-adjustment]
                    [-w WORKERS]

    Fits the Bacterial Rods model to a system of cells.

    options:
    -h, --help            show this help message and exit
    -i, --iteration ITERATION
                            Use existing output folder instead of creating new one
    --output-dir OUTPUT_DIR
                            Directory where to store results
    --skip-profiles       Skip plotting of profiles
    --skip-time-evolution
                            Skip plotting of the time evolution of costs
    --skip-snapshots      Skip plotting of snapshots and masks
    --skip-timings        Skip plotting of the timings
    --skip-mask-adjustment
                            Skip plotting of the adjusted masks
    --only-mask-adjustment
                            Only plot adjusted masks
    -w, --workers WORKERS
                            Number of threads to utilize

"""

import numpy as np
from glob import glob
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy as sp
import time
import argparse
import multiprocessing as mp
import cv2 as cv

import cr_mech_coli as crm
from cr_mech_coli import crm_fit

data_dir = Path("data/crm_divide/0001/")

crm.plotting.set_mpl_rc_params()


def adjust_masks(
    masks_data: list[np.ndarray[tuple[int, int], np.dtype[np.uint8]]],
    positions_all: list[np.ndarray[tuple[int, int, int], np.dtype[np.float32]]],
    iterations_data: list[int],
    container: crm.CellContainer,
):
    sim_iterations = np.array(container.get_all_iterations())
    sim_iterations_subset = np.array([sim_iterations[i] for i in iterations_data])
    sim_idents_all = container.get_all_identifiers()
    sim_daughter_map = container.get_daughter_map()

    # 0. Map data masks such that colors for non-dividing cells align
    #    and cells which have divided obtain their own new colorvalue
    # 1. Define parent map for data
    # 2. Get parent map from simulation
    # 3. Map data colors to simulation idents
    #       We check if the cell is daughter or mother
    #       If mother   -> match directly
    #       If daughter -> get parent -> match parent -> decide which daughter it is
    # 4. Update ident_to_color and parent_map for simulation data in order to still obtain correct
    #    relations for colors even if the corresponding cells are not present in the simulation
    # 5. Convert data colors by using
    #    Data Color -> Sim Ident -> Sim Color

    # Mapping to give masks after iteration 7 new colors
    # such that they do not overlap with previous results.
    # WARNING MAGIC NUMBERS
    align_mask_data_color = {
        np.uint8(1): np.uint8(20),
        np.uint8(2): np.uint8(21),
        np.uint8(3): np.uint8(22),
        np.uint8(4): np.uint8(23),
        np.uint8(5): np.uint8(24),
        np.uint8(6): np.uint8(25),
        np.uint8(7): np.uint8(26),
        np.uint8(8): np.uint8(5),
        np.uint8(9): np.uint8(27),
        np.uint8(10): np.uint8(6),
    }
    align_mask_data_color_invert = {v: k for k, v in align_mask_data_color.items()}

    # Tranform the data masks with the above mapping
    # WARNING MAGIC NUMBERS
    masks_data_new = [np.array(m) for m in masks_data]
    for m in masks_data_new[8:]:
        colors = list(sorted(np.unique(m)))[1:]
        for c in colors:
            m[m == c] = align_mask_data_color[c]

    # WARNING MAGIC NUMBERS
    data_color_parent_map = {
        np.uint8(1): None,
        np.uint8(2): None,
        np.uint8(3): None,
        np.uint8(4): None,
        np.uint8(5): None,
        np.uint8(6): None,
        np.uint8(21): 1,
        np.uint8(23): 1,
        np.uint8(20): 2,
        np.uint8(22): 2,
        np.uint8(24): 3,
        np.uint8(26): 3,
        np.uint8(25): 4,
        np.uint8(27): 4,
    }
    data_color_daughter_map = {
        parent_color: [k for k, v in data_color_parent_map.items() if v == parent_color]
        for parent_color in data_color_parent_map.values()
        if parent_color is not None
    }

    data_color_to_ident = {
        np.uint8(1): crm.CellIdentifier.new_initial(0),
        np.uint8(2): crm.CellIdentifier.new_initial(1),
        np.uint8(3): crm.CellIdentifier.new_initial(2),
        np.uint8(4): crm.CellIdentifier.new_initial(3),
        np.uint8(5): crm.CellIdentifier.new_initial(4),
        np.uint8(6): crm.CellIdentifier.new_initial(5),
    }

    for ident in sim_idents_all:
        if ident in data_color_to_ident.values():
            continue
        sim_parent = container.get_parent(ident)
        if sim_parent is None:
            raise ValueError("Could not find parent.")

        # Obtain all daughters from the simulation
        sim_daughters = sim_daughter_map[sim_parent]
        # Ensure that the original ident is in the list
        assert ident in sim_daughters

        # Obtain the histories of the daughters
        daughter_hists = [container.get_cell_history(d)[0] for d in sim_daughters]
        # Extract the first iteration at which the daughters are present and data is there
        daughter_iters = [
            # WARNING MAGIC NUMBER                               \|/
            [k for k in hist.keys() if k in sim_iterations_subset[8:]]
            for hist in daughter_hists
        ]

        first_shared_iter = np.max([np.min(i) for i in daughter_iters])
        n_first_shared = np.argmin(first_shared_iter > sim_iterations)
        (n_first_shared_data,) = np.where(n_first_shared == np.array(iterations_data))[
            0
        ]

        # The parent color can be obtained from the CellIdentifier::Initial(n) value (plus 1)
        parent_color = sim_parent[0] + 1
        # From there, we can infer the possible daughter colors
        daughter_colors = data_color_daughter_map[parent_color]

        # Now we can also obtain the extracted daughter positions
        daughter_positions = [
            positions_all[n_first_shared_data][align_mask_data_color_invert[dc] - 1]
            for dc in daughter_colors
        ]

        # And compare these positions with the simulation data
        # The closest position will be the chosen mapping
        sim_position = container.get_cell_history(ident)[0][first_shared_iter].pos
        d1s = [np.linalg.norm(q - sim_position) for q in daughter_positions]
        d2s = [np.linalg.norm(q[::-1] - sim_position) for q in daughter_positions]
        if np.min(d1s) < np.min(d2s):
            i = np.argmin(d1s)
        else:
            i = np.argmin(d2s)

        daughter_color = daughter_colors[i]
        data_color_to_ident[daughter_color] = ident

    # We have now matched all CellIdentifiers which are present
    # in the simulation and also in the data masks. Now we will
    # go on to insert relations for the remaining colors which
    # are present in the data but not in the simulation.

    data_colors_all = np.unique(masks_data_new)
    for new_color in data_colors_all[1:]:
        if new_color not in data_color_to_ident.keys():
            # Obtain parent color and ident
            parent_color = data_color_parent_map[new_color]
            parent_ident = crm.CellIdentifier.new_initial(parent_color - 1)

            # Use CellContainer to create new ident and update map
            new_ident = container.add_ident_divided(parent_ident)
            data_color_to_ident[new_color] = new_ident

    parent_map = container.get_parent_map()
    color_to_cell = container.color_to_cell
    cell_to_color = container.cell_to_color

    # Finally we build a dictionary which can
    # convert every data_color to sim_color
    data_color_to_sim_color = {np.uint8(0): crm.counter_to_color(0)}
    for k, ident in data_color_to_ident.items():
        sim_color = cell_to_color[ident]
        data_color_to_sim_color[k] = sim_color

    new_masks = []
    for mask in masks_data_new:
        new_mask = np.array(
            [data_color_to_sim_color[c] for c in mask.reshape(-1)], dtype=np.uint8
        ).reshape((*mask.shape, 3))
        new_masks.append(new_mask)

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
        30.0,
        30.0,
        8.0,
        11.0,
        200.0,
        200.0,
    ],
    growth_rates_new=[
        (0.001152799, 0.001152799),
        (0.001410604, 0.001410604),
        (0.0018761827, 0.0018761827),
        (0.0016834959, 0.0016834959),
        (0, 0),
        (0, 0),
    ],
    show_progress=False,
):
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
            growth_rate_setter={"g1": g1, "g2": g2},
            spring_length_threshold=spring_length_threshold,
            neighbor_reduction=None,
        )
        for spring_length_threshold, pos, growth_rate, (g1, g2) in zip(
            spring_length_thresholds,
            initial_positions,
            growth_rates,
            growth_rates_new,
        )
    ]

    # define config
    config = settings.to_config()
    if show_progress:
        config.progressbar = "Run Simulation"
    container = crm.run_simulation_with_agents(config, agents)
    if show_progress:
        print()

    return container


ERROR_COST = 1e6


def objective_function(
    spring_length_thresholds_and_new_growth_rates,
    positions_all,
    settings,
    masks_data,
    iterations_data,
    parent_penalty=0.5,
    return_all=False,
    return_times=False,
    show_progressbar=False,
    print_costs=True,
):
    times = [(time.perf_counter_ns(), "Start")]

    def update_time(message):
        if return_times:
            now = time.perf_counter_ns()
            times.append((now, message))

    spring_length_thresholds = spring_length_thresholds_and_new_growth_rates[:4]
    growth_rates_new = [
        *np.array(spring_length_thresholds_and_new_growth_rates[4:]).reshape((-1, 2)),
        # These should not come into effect at all
        (0.0, 0.0),
        (0.0, 0.0),
    ]

    try:
        container = predict(
            positions_all[0],
            settings,
            spring_length_thresholds=[*spring_length_thresholds, 200.0, 200.0],
            growth_rates_new=growth_rates_new,
            show_progress=show_progressbar,
        )
    except ValueError or KeyError as e:
        if return_all:
            raise e
        return ERROR_COST
    iterations_simulation = np.array(container.get_all_iterations()).astype(int)

    update_time("Predict")

    try:
        new_masks, parent_map, cell_to_color, color_to_cell = adjust_masks(
            masks_data, positions_all, iterations_data, container
        )
    except Exception as e:
        if print_costs:
            print(f"Error -> f(x)={ERROR_COST} error: {e}")
        return ERROR_COST

    update_time("Masks\n(Adjust)")

    iters_filtered = np.array([iterations_simulation[i] for i in iterations_data])
    masks_predicted = [
        crm.render_mask(
            container.get_cells_at_iteration(iter),
            cell_to_color,
            settings.constants.domain_size,
            render_settings=crm.RenderSettings(pixel_per_micron=1),
        )
        for iter in tqdm(
            iterations_simulation if return_all else iters_filtered,
            total=len(iterations_simulation if return_all else iters_filtered),
            desc="Render predicted Masks",
            disable=not show_progressbar,
        )
    ]

    update_time("Masks\n(Render)")

    # If we return all we need to filter the generated masks
    if return_all:
        mask_iterator = zip(
            [masks_predicted[iter] for iter in iterations_data], new_masks
        )
    # Otherwise we can use the whole list
    else:
        mask_iterator = zip(masks_predicted, new_masks)

    diff_masks = np.array(
        [
            crm.parents_diff_mask(
                m1,
                m2,
                color_to_cell,
                parent_map,
                parent_penalty,
            )
            for m1, m2 in mask_iterator
        ]
    )

    penalties = np.sum(diff_masks, axis=(1, 2))

    update_time("Compare")

    if return_all:
        return (
            new_masks,
            parent_map,
            cell_to_color,
            color_to_cell,
            container,
            masks_predicted,
        )

    if return_times:
        return times

    n_cells = len(container.get_cells_at_iteration(iterations_simulation[-1]))
    cost = np.sum(penalties) * (1 + (n_cells - 10) ** 2) ** 0.5

    if print_costs:
        print(
            f"f(x)={cost:>10.1f}  Final Cells: {n_cells:2} Penalties: {np.sum(penalties):<10.1f}"
        )
    return cost


def preprocessing(n_masks=None):
    if n_masks is None:
        files_images = sorted(glob(str(data_dir / "images/*")))
        files_masks = sorted(glob(str(data_dir / "masks/*.csv")))
    else:
        files_images = list(sorted(glob(str(data_dir / "images/*"))))[:n_masks]
        files_masks = list(sorted(glob(str(data_dir / "masks/*.csv"))))[:n_masks]
    masks = [np.loadtxt(fm, delimiter=",", dtype=np.uint8) for fm in files_masks]
    iterations_data = np.array([int(s[-10:-4]) for s in files_images])
    iterations_data = iterations_data - np.min(iterations_data)

    settings = crm_fit.Settings.from_toml(data_dir / "settings.toml")
    n_vertices = settings.constants.n_vertices

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
            colors_all.append(colors)
        except ValueError as e:
            print("Encountered Error during extraction of positions:")
            print(filename)
            print(e)
            print("Omitting this particular result.")

    settings.constants.n_saves = max(iterations_data)

    domain_height = settings.domain_height
    for n, p in enumerate(positions_all):
        positions_all[n] = np.append(
            p,
            domain_height / 2 + np.zeros((*p.shape[:2], 1)),
            axis=2,
        ).astype(np.float32)

    return masks, positions_all, settings, iterations_data


def plot_mask_adjustment(
    output_dir, masks_data, positions_all, settings, iterations_data
):
    spring_length_thresholds = [15] * 4
    new_growth_rates = [0.001] * 8
    x0 = [
        *spring_length_thresholds,
        *new_growth_rates,
    ]

    args = (
        positions_all,
        settings,
        masks_data,
        iterations_data,
        0.5,
    )

    (
        masks_adjusted,
        parent_map,
        cell_to_color,
        color_to_cell,
        container,
        masks_predicted,
    ) = objective_function(x0, *args, return_all=True, show_progressbar=True)

    (output_dir / "mask_adjustments").mkdir(parents=True, exist_ok=True)
    for mask_predicted, mask_adjusted, mask_data, mask_iter in tqdm(
        zip(
            [masks_predicted[i] for i in iterations_data],
            masks_adjusted,
            masks_data,
            iterations_data,
        ),
        total=len(masks_adjusted),
        desc="Plot Adjustments",
    ):
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        axs[0, 0].set_axis_off()
        axs[0, 0].set_title("Mask Data")
        axs[0, 1].set_axis_off()
        axs[0, 1].set_title("Mask Predicted")
        axs[1, 0].set_axis_off()
        axs[1, 0].set_title("Mask Adjusted")
        axs[1, 1].set_axis_off()
        axs[1, 1].set_title("Diff")

        diff = crm.parents_diff_mask(
            mask_predicted, mask_adjusted, color_to_cell, parent_map, 0.5
        )

        axs[0, 0].imshow(mask_data)

        def ident_to_text(ident):
            try:
                return f"D({ident[1]})"
            except:
                return f"I({ident[0]})"

        colors = list(sorted(np.unique(mask_data)))[1:]
        for n, v in enumerate(colors):
            x, y = np.where(mask_data == v)
            pos = np.mean([x, y], axis=1)
            axs[0, 0].text(
                pos[1], pos[0], n + 1, color="white", fontfamily="sans-serif", size=10
            )

        for k, v in container.cell_to_color.items():
            x, y = np.where(np.all(mask_predicted == v, axis=2))
            pos = np.mean([x, y], axis=1)
            axs[0, 1].text(
                pos[1],
                pos[0],
                ident_to_text(k),
                color="white",
                fontfamily="sans-serif",
                size=10,
            )

        axs[0, 1].imshow(mask_predicted)
        axs[1, 0].imshow(mask_adjusted)

        for k, v in cell_to_color.items():
            x, y = np.where(np.all(mask_adjusted == v, axis=2))
            pos = np.mean([x, y], axis=1)
            axs[1, 0].text(
                pos[1],
                pos[0],
                ident_to_text(k),
                color="white",
                fontfamily="sans-serif",
                size=10,
            )

        axs[1, 1].imshow(1 - diff, cmap="Grays")

        fig.savefig(output_dir / f"mask_adjustments/{mask_iter:06}.png")
        plt.close(fig)


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
    fig.savefig(output_dir / "time-evolution.pdf")
    fig.savefig(output_dir / "time-evolution.png")
    plt.close(fig)


def __optimize_around_single(params, param_single, n, args):
    all_params = np.array([*params[:n], param_single, *params[n:]])
    return objective_function(all_params, *args, print_costs=False)


def __calculate_single_cost(n, p, parameters, bounds, args):
    index = np.arange(len(parameters)) != n
    x0 = np.array(parameters)[index]
    bounds_reduced = np.array(bounds)[index]

    assert len(x0) + 1 == len(parameters)
    assert len(bounds_reduced) + 1 == len(bounds)

    res = sp.optimize.minimize(
        __optimize_around_single,
        x0=x0,
        method="Nelder-Mead",
        bounds=bounds_reduced,
        args=(p, n, args),
        options={
            "disp": False,
            "maxiter": 12,
            "maxfev": 12,
        },
    )

    return res.fun


def plot_profiles(
    parameters: np.ndarray,
    bounds,
    labels: list,
    final_cost: float,
    args,
    output_dir,
    n_workers: int,
):
    from itertools import repeat

    pool = mp.Pool(n_workers)

    n_samples = 60
    b_low = np.array(bounds)[:, 0]
    b_high = np.array(bounds)[:, 1]
    n_param = np.repeat([np.arange(len(parameters))], n_samples, axis=0)
    samples = np.linspace(b_low, b_high, n_samples)

    arglist = tqdm(
        zip(
            n_param.flatten(),
            samples.flatten(),
            repeat(parameters),
            repeat(bounds),
            repeat(args),
        ),
        total=int(np.prod(n_param.shape)),
        desc="Calculating Costs",
    )

    costs = pool.starmap(__calculate_single_cost, arglist)
    costs = np.array(costs).reshape((n_samples, len(parameters)))

    for n, p, costs_ind, samples_ind in zip(
        range(len(parameters)), parameters, costs.T, samples.T
    ):
        fig, ax = plt.subplots(figsize=(8, 8))
        crm.configure_ax(ax)

        # Add previously calculated results
        x = np.array([p, *samples_ind])
        y = np.array([final_cost, *costs_ind])

        # Filter out values that indicate an error
        x = x[y < ERROR_COST]
        y = y[y < ERROR_COST]

        # Sort entries by value of the parameter
        inds = np.argsort(x)
        x = x[inds]
        y = y[inds]

        ax.plot(x, y, c=crm.plotting.COLOR3, marker="x")
        ax.scatter([parameters[n]], [final_cost], c=crm.plotting.COLOR5)
        ax.set_title(labels[n])
        odir = output_dir / "profiles"
        odir.mkdir(parents=True, exist_ok=True)
        fig.savefig(odir / f"profile-{n:06}.png")
        plt.close(fig)


def plot_timings(
    parameters,
    positions_all,
    settings,
    masks_data,
    iterations_data,
    output_dir,
    n_samples: int = 3,
):
    times = []
    for _ in tqdm(range(n_samples), total=n_samples, desc="Measure Timings"):
        times.append(
            objective_function(
                parameters,
                positions_all,
                settings,
                masks_data,
                iterations_data,
                parent_penalty=0.5,
                return_times=True,
            )
        )

    data = np.array(
        [[times[i][j][0] for j in range(len(times[0]))] for i in range(len(times))]
    )
    data = (data[:, 1:] - data[:, :-1]) / 1e9
    mean = np.mean(data, axis=0)
    ind = np.argsort(mean)[::-1]
    mean = mean[ind]
    dmean = np.std(data, axis=0)[ind]
    labels = np.array([t[1] for t in times[0][1:]])[ind]
    perc = mean / np.sum(mean)
    dperc = (
        (dmean / np.sum(mean)) ** 2 + (np.sum(dmean) * mean / np.sum(mean) ** 2) ** 2
    ) ** 0.5

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_ylim(0, np.max(mean) * 1.15)
    crm.configure_ax(ax)
    b = ax.bar(labels, mean, color=crm.plotting.COLOR3)
    ax.bar_label(
        b,
        [f"{100 * p:.2f}%\n±{100 * dp:.2f}%" for p, dp in zip(perc, dperc)],
        label_type="edge",
        color=crm.plotting.COLOR5,
        weight="bold",
    )
    ax.set_ylabel("Time [s]")
    fig.savefig(output_dir / "timings.pdf")
    fig.savefig(output_dir / "timings.png")


def calculate_single(args):
    return (args[0], objective_function(*args))


def run_optimizer(
    spring_length_thresholds_and_new_growth_rates,
    bounds,
    output_dir,
    iteration,
    args,
    n_workers,
):
    # Try loading data
    if iteration is not None:
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
            maxiter=100,
            popsize=15,
            mutation=(0.0, 1.5),
            recombination=0.6,
            tol=0.0001,
            workers=n_workers,
            updating="deferred",
            polish=True,
            init="latinhypercube",
            strategy="best1bin",
        )
        final_parameters = res.x
        final_cost = res.fun
        np.savetxt(output_dir / "optimize_result.csv", [*final_parameters, final_cost])

    return final_parameters, final_cost


def plot_snapshots(
    iterations_data,
    masks_predicted,
    masks_adjusted,
    output_dir,
    color_to_cell,
    parent_map,
):
    (output_dir / "masks_predicted").mkdir(parents=True, exist_ok=True)
    (output_dir / "masks_adjusted").mkdir(parents=True, exist_ok=True)
    (output_dir / "masks_diff").mkdir(parents=True, exist_ok=True)
    for n, m in enumerate(masks_predicted):
        cv.imwrite(f"{output_dir}/masks_predicted/{n:06}.png", m)
    for n, m2 in zip(iterations_data, masks_adjusted):
        m1 = masks_predicted[n]
        cv.imwrite(f"{output_dir}/masks_adjusted/{n:06}.png", m2)
        diff = (
            crm.parents_diff_mask(m1, m2, color_to_cell, parent_map, 0.5) * 255
        ).astype(np.uint8)
        cv.imwrite(f"{output_dir}/masks_diff/{n:06}.png", diff)


def crm_divide_main():
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
        "--skip-profiles",
        action="store_true",
        help="Skip plotting of profiles",
    )
    parser.add_argument(
        "--skip-time-evolution",
        action="store_true",
        help="Skip plotting of the time evolution of costs",
    )
    parser.add_argument(
        "--skip-snapshots",
        action="store_true",
        help="Skip plotting of snapshots and masks",
    )
    parser.add_argument(
        "--skip-timings",
        action="store_true",
        help="Skip plotting of the timings",
    )
    parser.add_argument(
        "--skip-mask-adjustment",
        action="store_true",
        help="Skip plotting of the adjusted masks",
    )
    parser.add_argument(
        "--only-mask-adjustment",
        action="store_true",
        help="Only plot adjusted masks",
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

    masks_data, positions_all, settings, iterations_data = preprocessing()

    if not pyargs.skip_mask_adjustment or pyargs.only_mask_adjustment:
        plot_mask_adjustment(
            output_dir, masks_data, positions_all, settings, iterations_data
        )
        if pyargs.only_mask_adjustment:
            exit()

    spring_length_thresholds = [
        9.405841188088112759e00,
        1.103879179742345329e01,
        9.277040994277374608e00,
        7.600778468661159692e00,
    ]
    new_growth_rates = [
        2.186928453188154847e-03,
        2.186928453188154847e-03,
        4.834987621055546192e-04,
        4.834987621055546192e-04,
        1.669060863394238470e-03,
        1.669060863394238470e-03,
        2.300937616285135622e-03,
        2.300937616285135622e-03,
    ]
    x0 = [
        *spring_length_thresholds,
        *new_growth_rates,
    ]
    bounds = [(5, 12)] * 4 + [(0.0000, 0.004)] * 8
    parent_penalty = 0.5
    args = (
        positions_all,
        settings,
        masks_data,
        iterations_data,
        parent_penalty,
    )

    final_parameters, final_cost = run_optimizer(
        x0,
        bounds,
        output_dir,
        pyargs.iteration,
        args,
        n_workers,
    )

    (
        masks_adjusted,
        parent_map,
        _,
        color_to_cell,
        container,
        masks_predicted,
    ) = objective_function(
        final_parameters, *args, return_all=True, show_progressbar=True
    )

    if not pyargs.skip_snapshots:
        plot_snapshots(
            iterations_data,
            masks_predicted,
            masks_adjusted,
            output_dir,
            color_to_cell,
            parent_map,
        )

    if not pyargs.skip_time_evolution:
        plot_time_evolution(
            masks_predicted,
            masks_adjusted,
            color_to_cell,
            parent_map,
            container.get_all_iterations(),
            iterations_data,
            settings,
            output_dir,
        )

    if not pyargs.skip_profiles:
        labels = [
            "Division Length 0",
            "Division Length 1",
            "Division Length 2",
            "Division Length 3",
            "Growth Rate 0-0",
            "Growth Rate 0-1",
            "Growth Rate 1-0",
            "Growth Rate 1-1",
            "Growth Rate 2-0",
            "Growth Rate 2-1",
            "Growth Rate 3-0",
            "Growth Rate 3-1",
        ]
        plot_profiles(
            final_parameters,
            bounds,
            labels,
            final_cost,
            args,
            output_dir,
            n_workers,
        )

    if not pyargs.skip_timings:
        plot_timings(
            final_parameters,
            positions_all,
            settings,
            masks_data,
            iterations_data,
            output_dir,
        )
