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
# from cr_mech_coli.crm_divide.crm_divide_rs import adjust_masks

data_dir = Path("data/crm_divide/0001/")

crm.plotting.set_mpl_rc_params()


def adjust_masks(
    masks_data: list[np.ndarray[tuple[int, int], np.dtype[np.uint8]]],
    positions_all: list[np.ndarray[tuple[int, int, int], np.dtype[np.float32]]],
    mask_iters: list[int],
    container: crm.CellContainer,
    settings: crm_fit.Settings,
    show_progress=False,
):
    sim_iterations = np.array(container.get_all_iterations())
    sim_iterations_subset = np.array([sim_iterations[i] for i in mask_iters])
    sim_idents_initial = container.get_cells_at_iteration(sim_iterations[0]).keys()
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
    align_mask_data_color = {
        1: 20,
        2: 21,
        3: 22,
        4: 23,
        5: 24,
        6: 25,
        7: 26,
        8: 5,
        9: 27,
        10: 6,
    }
    align_mask_data_color_invert = {v: k for k, v in align_mask_data_color.items()}

    # Tranform the data masks with the above mapping
    masks_data_new = []
    for m in masks_data[7:]:
        new_mask = m
        colors = list(sorted(np.unique(m)))[1:]
        for c in colors:
            new_mask[m == c] = align_mask_data_color[c]
        masks_data_new.append(new_mask)

    masks_data[7:] = masks_data_new

    data_color_parent_map = {
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
    data_color_daughter_map = {
        parent_color: [k for k, v in data_color_parent_map.items() if v == parent_color]
        for parent_color in data_color_parent_map.values()
        if parent_color is not None
    }

    data_color_to_ident = {
        1: crm.CellIdentifier.new_initial(0),
        2: crm.CellIdentifier.new_initial(1),
        3: crm.CellIdentifier.new_initial(2),
        4: crm.CellIdentifier.new_initial(3),
        5: crm.CellIdentifier.new_initial(4),
        6: crm.CellIdentifier.new_initial(5),
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
            [k for k in hist.keys() if k in sim_iterations_subset]
            for hist in daughter_hists
        ]
        first_shared_iter = np.max([np.min(i) for i in daughter_iters])
        n_first_shared = np.argmin(first_shared_iter > sim_iterations)
        (n_first_shared_data,) = np.where(n_first_shared == np.array(mask_iters))[0]

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

        # fig, ax = plt.subplots(figsize=(15, 12))
        # ax.imshow(masks_data[n_first_shared_data][::-1])
        # ax.set_xlim(0, settings.constants.domain_size[0])
        # ax.set_ylim(0, settings.constants.domain_size[1])
        # ax.plot(
        #     sim_position[:, 0], sim_position[:, 1], color="red", marker="x", alpha=0.5
        # )

        # print(ident)
        # m = masks_data[n_first_shared_data]
        # for color in np.unique(m)[1:]:
        #     x, y = np.where(m == color)
        #     pos_mean = np.mean([x, y], axis=1)
        #     color_original = align_mask_data_color_invert[color]
        #     color_parent = data_color_parent_map[color]
        #     color_parent = "" if color_parent is None else color_parent
        #     ax.text(
        #         pos_mean[1],
        #         m.shape[0] - pos_mean[0],
        #         f"{color}:{color_original}:{color_parent}",
        #         color="white",
        #     )

        # for j, pi in enumerate(daughter_positions):
        #     ax.plot(pi[:, 0], pi[:, 1], color="blue" if i == j else "gray", marker="+")
        # plt.show()
        # plt.close(fig)

    # We have now matched all CellIdentifiers which are present
    # in the simulation and also in the data masks. Now we will
    # go on to insert relations for the remaining colors which
    # are present in the data but not in the simulation.

    parent_map = container.get_parent_map()
    color_to_cell = container.color_to_cell
    cell_to_color = container.cell_to_color

    ident_counter = len(data_color_to_ident) - len(sim_idents_initial) + 1
    data_colors_all = np.unique(masks_data)
    for new_color in data_colors_all[1:]:
        if new_color not in data_color_to_ident.keys():
            # Obtain parent color and ident
            parent_color = data_color_parent_map[new_color]
            parent_ident = crm.CellIdentifier.new_initial(parent_color - 1)

            # Create new artificial ident and obtain color for ident
            new_ident = crm.CellIdentifier(crm.VoxelPlainIndex(0), ident_counter)
            sim_color = crm.counter_to_color(ident_counter + len(sim_idents_initial))

            # Extend parent map
            ident_counter += 1
            data_color_to_ident[new_color] = new_ident
            parent_map[new_ident] = parent_ident
            color_to_cell[sim_color] = new_ident
            cell_to_color[new_ident] = sim_color

    # Finally we build a dictionary which can
    # convert every data_color to sim_color
    data_color_to_sim_color = {0: crm.counter_to_color(0)}
    for k, ident in data_color_to_ident.items():
        sim_color = cell_to_color[ident]
        data_color_to_sim_color[k] = sim_color

    new_masks = []
    for mask in masks_data:
        new_mask = np.array(
            [data_color_to_sim_color[c] for c in mask.reshape(-1)], dtype=np.uint8
        ).reshape((*mask.shape, 3))
        new_masks.append(new_mask)

    print(len(parent_map))
    print(len(color_to_cell))
    print(len(cell_to_color))
    exit()

    return None

    # Define mappings for colors of the data masks at each iteration
    color_mappings_at_iteration = [
        {i: crm.CellIdentifier.new_initial(i) for i in range(1, 7)}
    ] * 8
    color_mappings_at_iteration += [
        {i: crm.CellIdentifier.new_initial(i - 1) for i in range(1, 3)}
    ] * 5

    def get_closest_pos(n, pos_cell):
        min_ind = 0
        min_pos = np.array(0)
        min_dist = np.inf
        for i, p in enumerate(positions_all[n]):
            d1 = np.linalg.norm(p - pos_cell)
            d2 = np.linalg.norm(p - pos_cell[::-1])

            if d1 < min_dist:
                min_pos = p
                min_ind = i
                min_dist = d1
            if d2 < min_dist:
                min_pos = p
                min_ind = i
                min_dist = d2

        return (min_ind, min_pos, min_dist)

    for n in range(8, 8 + 5):
        iter = container.get_all_iterations()[mask_iters[n]]
        cells = container.get_cells_at_iteration(iter)

        data_color_to_ident = {}
        for ident, (c, sim_parent) in cells.items():
            pos = c.pos
            # This index is now the color of the unaltered data mask
            (min_ind, min_pos, min_dist) = get_closest_pos(n, pos)
            # We can thus map the identifier of the cell to this data-color
            data_color_to_ident[min_ind] = ident
            print(min_ind)

        print(data_color_to_ident.keys(), color_mappings_at_iteration[n].keys())
        color_mappings_at_iteration[n] = (
            color_mappings_at_iteration[n] | data_color_to_ident
        )
        print(len(color_mappings_at_iteration[n]))

    for mapping in color_mappings_at_iteration:
        print(len(mapping))

    exit()

    # 1. Identify new cells from simulation
    # 2. Link new cells with colors from later masks
    # 3. Extend cell_to_color and parent map

    # Map colors such that the cells which have not divided match with the previous masks
    # Mask color
    # 8 -> 5
    # 10 -> 6
    # In order to avoid collisions we also map divided cells to colors of value >= 20
    align_mask_data_color = {
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
        for k, v in align_mask_data_color.items():
            m[m == k] = v
        return m

    x = np.array([len(np.unique(m)) - 1 for m in masks_data])
    n_divide = np.argmin(x < x[-1])

    masks_data = [
        m if n < n_divide else map_colors(m) for n, m in enumerate(masks_data)
    ]

    # Map colors to Idents which are there initially
    # These idents do not change
    # mask_color_to_ident = {i: crm.CellIdentifier.new_initial(i) for i in range(1, 7)}
    # parent_map = {mask_color_to_ident[k]: None for k in range(1, 7)}

    mask_color_to_cell = {}

    # Calculate the average pixel for each color
    color_means = [
        {c: np.mean(np.where(m == c), axis=1) for c in np.unique(m)} for m in masks_data
    ]
    for id in sim_idents_all:
        if id not in sim_idents_initial:
            # Get history of cell
            cell_hist, parent_id = container.get_cell_history(id)
            # Obtain the first iteration at which the new cell appears
            first_iter = min(cell_hist.keys())
            # Obtain the very first position of this cell
            # and convert the position to pixel units
            first_pos = cell_hist[first_iter].pos
            first_pos = crm.convert_cell_pos_to_pixels(
                first_pos, settings.constants.domain_size, masks_data[0].shape
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
    for c in [np.unique(m) for m in masks_data]:
        for ci in c:
            assert ci in parent_map_colors.keys() or ci == 0

    # Create new masks with updated colors
    new_masks = []
    if show_progress:
        iterator = tqdm(masks_data, total=len(masks_data), desc="Adjusting Data Masks")
    else:
        iterator = masks_data
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
    if show_progress:
        config.progressbar = "Run Simulation"
    container = crm.run_simulation_with_agents(config, agents)
    if show_progress:
        print()

    return container


def objective_function(
    spring_length_thresholds_and_new_growth_rates,
    positions_all,
    settings,
    masks_data,
    mask_iters,
    iterations_data,
    parent_penalty=0.5,
    return_all=False,
    return_times=False,
    error_cost=10.0,
    show_progressbar=False,
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
            positions_all[0],
            settings,
            spring_length_thresholds=[*spring_length_thresholds, 200.0, 200.0],
            growth_rate_distrs=[(g, 0) for g in new_growth_rates],
            show_progress=show_progressbar,
        )
    except ValueError or KeyError as e:
        if return_all:
            raise e
        return error_cost
    iterations_simulation = np.array(container.get_all_iterations()).astype(int)

    update_time("Prediction")

    # try:
    new_masks, parent_map, cell_to_color, color_to_cell = adjust_masks(
        masks_data, positions_all, mask_iters, container, settings
    )
    # except:
    #     return error_cost

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0, 0].imshow(new_masks[-1])
    axs[0, 1].imshow(masks_data[-1])

    m = masks_data[-1]
    colors = np.unique(m)
    for n, v in enumerate(colors):
        x, y = np.where(m == v)
        pos = np.mean([x, y], axis=1)
        axs[0, 1].text(
            pos[1], pos[0], n, color="gray", fontfamily="sans-serif", size=15
        )

    # axs[1,0].imshow(masks_predicted[-1])
    # axs[1,1].imshow(masks_data[-1])
    fig.savefig("tmp2.png")

    exit()

    masks_predicted = [
        crm.render_mask(
            container.get_cells_at_iteration(iter),
            cell_to_color,
            settings.constants.domain_size,
            render_settings=crm.RenderSettings(pixel_per_micron=1),
        )
        for iter in tqdm(
            iterations_simulation if return_all else iterations_data,
            total=len(iterations_simulation if return_all else iterations_data),
            desc="Render predicted Masks",
            disable=not show_progressbar,
        )
    ]

    update_time("Masks")

    def mask_iterator():
        if return_all:
            for iter, new_mask in zip(iterations_data, new_masks):
                yield new_mask, masks_predicted[iter]

        else:
            for m1, m2 in zip(new_masks, masks_predicted):
                yield m1, m2

    penalties = [
        crm.penalty_area_diff_account_parents(
            m1,
            m2,
            color_to_cell,
            parent_map,
            parent_penalty,
        )
        for m1, m2 in mask_iterator()
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

    n_cells = len(container.get_cells_at_iteration(iterations_simulation[-1]))

    cost = np.sum(penalties) * (1 + (n_cells - 10) ** 2) ** 0.5

    if return_times:
        return times

    print(f"f(x)={cost:12.7}  Final Cells: {n_cells}")
    return cost


def preprocessing(n_masks=None):
    if n_masks is None:
        files_images = sorted(glob(str(data_dir / "images/*")))
        files_masks = sorted(glob(str(data_dir / "masks/*.csv")))
    else:
        files_images = list(sorted(glob(str(data_dir / "images/*"))))[:n_masks]
        files_masks = list(sorted(glob(str(data_dir / "masks/*.csv"))))[:n_masks]
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

    domain_height = settings.domain_height
    for n, p in enumerate(positions_all):
        positions_all[n] = np.append(
            p,
            domain_height / 2 + np.zeros((*p.shape[:2], 1)),
            axis=2,
        ).astype(np.float32)

    return masks, positions_all, settings, iterations_all, mask_iters


def test_adjust_masks():
    masks_data, positions_all, settings, iterations_data, mask_iters = preprocessing()

    spring_length_thresholds = [9] * 4
    new_growth_rates = [
        0.001152799,
        0.001410604,
        0.0018761827,
        0.0016834959,
    ]
    x0 = [
        *spring_length_thresholds,
        *new_growth_rates,
    ]

    args = (
        positions_all,
        settings,
        masks_data,
        mask_iters,
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
        penalties,
    ) = objective_function(x0, *args, return_all=True, show_progressbar=True)

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
        masks_predicted[0], masks_adjusted[0], color_to_cell, parent_map, 0.5
    )

    axs[0, 0].imshow(masks_data[0])

    m = masks_data[0]
    colors = np.unique(m)
    for n, v in enumerate(colors):
        x, y = np.where(m == v)
        pos = np.mean([x, y], axis=1)
        axs[0, 0].text(
            pos[1], pos[0], n, color="gray", fontfamily="sans-serif", size=20
        )

    m = masks_predicted[0]
    for k, v in container.cell_to_color.items():
        x, y = np.where(np.all(m == v, axis=2))
        pos = np.mean([x, y], axis=1)
        axs[0, 1].text(
            pos[1], pos[0], k, color="white", fontfamily="sans-serif", size=10
        )

    axs[0, 1].imshow(masks_predicted[0])

    m = masks_adjusted[0]
    for k, v in cell_to_color.items():
        x, y = np.where(np.all(m == v, axis=2))
        pos = np.mean([x, y], axis=1)
        axs[1, 0].text(
            pos[1], pos[0], k, color="white", fontfamily="sans-serif", size=10
        )

    axs[1, 0].imshow(masks_adjusted[0])
    axs[1, 1].imshow(1 - diff, cmap="Grays")

    fig.savefig("tmp.png")
    exit()


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


def optimize_around_single(params, param_single, n, args):
    all_params = np.array([*params[:n], param_single, *params[n + 1 :]])
    return objective_function(all_params, *args)


def plot_profiles(
    parameters,
    bounds,
    final_cost: float,
    args,
    output_dir,
    n_workers: int,
):
    for n, (p, (b_lower, b_upper)) in tqdm(
        enumerate(zip(parameters, bounds)),
        total=len(parameters),
        desc="Plotting Profiles",
    ):
        costs = []
        index = np.arange(len(parameters)) != n
        dx = (b_upper - b_lower) / 50
        n_samples = 10
        # x = (p + np.arange(-3, 3) * dx)[np.arange(-3, 3) != 0]
        x = np.linspace(
            max(p - dx, b_lower), min(p + dx, b_upper), n_samples, endpoint=True
        )
        for xi in x:
            x0 = np.array(parameters)[index]
            bounds_reduced = np.array(bounds)[index]
            assert len(x0) + 1 == len(parameters)
            assert len(bounds_reduced) + 1 == len(bounds)

            res = sp.optimize.differential_evolution(
                optimize_around_single,
                x0=x0,
                bounds=bounds_reduced,
                args=(xi, n, args),
                disp=False,
                maxiter=50,
                popsize=20,
                mutation=(0.6, 1),
                recombination=0.5,
                workers=n_workers,
                updating="deferred",
            )

            costs.append(res.fun)

        fig, ax = plt.subplots(figsize=(8, 8))
        crm.configure_ax(ax)
        x = np.array([parameters[n], *x])
        y = np.array([final_cost, *costs])
        inds = np.argsort(x)
        x = x[inds]
        y = y[inds]

        ax.plot(x, y, c=crm.plotting.COLOR3, marker="x")
        ax.scatter([parameters[n]], [final_cost], c=crm.plotting.COLOR5)
        fig.savefig(output_dir / f"profile-{n:010}.png")
        plt.close(fig)


def plot_timings(
    parameters,
    positions_all,
    settings,
    masks_data,
    mask_iters,
    iterations_data,
    output_dir,
    n_samples: int = 2,
):
    times = []
    for _ in tqdm(range(n_samples), total=n_samples, desc="Measure Timings"):
        times.append(
            # [("p0", 1 * n), ("p1", 2 * n), ("p2", 3 * n)]
            objective_function(
                parameters,
                positions_all,
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
    ind = np.argsort(mean)[::-1]
    mean = mean[ind]
    labels = np.array([t[1] for t in times[0][1:]])[ind]
    perc = mean / np.sum(mean)

    fig, ax = plt.subplots(figsize=(8, 8))
    crm.configure_ax(ax)
    b = ax.bar(labels, mean, color=crm.plotting.COLOR3)
    ax.bar_label(
        b,
        [f"{100 * p:5.3}%" for p in perc],
        label_type="center",
        color=crm.plotting.COLOR5,
        weight="bold",
    )
    # ax.set_yscale("log")
    ax.set_ylabel("Time [ms]")
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
        # lhs = sp.stats.qmc.LatinHypercube(
        #     d=len(spring_length_thresholds_and_new_growth_rates)
        # )
        # b = np.array(bounds)
        # samples = b[:, 0] + lhs.random(n=50) * (b[:, 1] - b[:, 0])

        # pool_args = [(s, *args) for s in samples]
        # costs = process_map(
        #     calculate_single,
        #     pool_args,
        #     max_workers=n_workers,
        #     desc="Finding Global Optimum",
        # )
        # final_parameters, final_cost = min(costs, key=lambda x: x[1])

        res = sp.optimize.differential_evolution(
            objective_function,
            x0=spring_length_thresholds_and_new_growth_rates,
            bounds=bounds,
            args=args,
            disp=True,
            maxiter=200,
            popsize=20,
            mutation=(0.3, 1.8),
            recombination=0.25,
            workers=n_workers,
            updating="deferred",
            polish=True,
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
    test_adjust_masks()

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

    masks_data, positions_all, settings, iterations_data, mask_iters = preprocessing()

    spring_length_thresholds = [9] * 4
    new_growth_rates = [
        0.001152799,
        0.001410604,
        0.0018761827,
        0.0016834959,
    ]
    x0 = [
        *spring_length_thresholds,
        *new_growth_rates,
    ]
    bounds = [(4.3, 10)] * 4 + [(0.001, 0.002)] * 4
    parent_penalty = 0.5
    args = (
        positions_all,
        settings,
        masks_data,
        mask_iters,
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
        cell_to_color,
        color_to_cell,
        container,
        masks_predicted,
        penalties,
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
        plot_profiles(
            final_parameters,
            bounds,
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
            mask_iters,
            iterations_data,
            output_dir,
        )
