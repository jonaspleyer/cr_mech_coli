import numpy as np
from glob import glob
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

import cr_mech_coli as crm
from cr_mech_coli import crm_fit

data_dir = Path("data/crm_divide/0001/")


def adjust_masks(masks, mask_iters, container: crm.CellContainer, settings):
    idents = container.cell_to_color.keys()
    iterations = container.get_all_iterations()
    idents_initial = container.get_cells_at_iteration(iterations[0]).keys()

    cell_to_color = container.cell_to_color
    parent_map = container.parent_map

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

    cell_to_mask_color = {}

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
            # print(first_pos)
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
            n_ind = np.min(np.where(n_iter < mask_iters))

            # We can now obtain the mean values of the children
            child_means = [color_means[n_ind][c] for c in child_colors]
            # Now we calculate the distance between the mean position of our agent
            # and the position of the candidates which is identical to the children
            # of the parent of the agent
            distances = [np.sum((pos_mean - cm) ** 2) ** 0.5 for cm in child_means]

            # We choose the child with the smaller distance as the correct one
            child_color = child_colors[np.argmin(distances)]
            cell_to_mask_color[id] = child_color

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

    return masks

    x = np.array([len(np.unique(m)) - 1 for m in masks])
    n_divide = np.argmin(x < x[-1])

    def prepare_mask(m):
        #
        map_colors = {
            1: 7,
            2: 8,
            3: 9,
            4: 10,
            5: 11,
            6: 13,
            7: 12,
        }
        colors_adjust1 = {
            8: 100,
            9: 101,
            10: 102,
        }
        colors_adjust2 = {
            100: 5,
            101: 14,
            102: 6,
        }

        for k in colors_adjust1.keys():
            m[m == k] = colors_adjust1[k]
        for k in sorted(map_colors.keys(), reverse=True):
            m[m == k] = map_colors[k]
        for k in colors_adjust2.keys():
            m[m == k] = colors_adjust2[k]

        return m

    masks_new = [m if n < n_divide else prepare_mask(m) for n, m in enumerate(masks)]
    for m in masks_new:
        print(np.unique(m))
        print(len(np.unique(m)))


def predict(
    initial_positions,
    settings,
    radius=0.5,
    strength=0.1,
    bound=10,
    cutoff=100,
    en=2.0,
    em=1.0,
    diffusion_constant=0.0,
    spring_tension=3.0,
    rigidity=10.0,
    spring_length=3.0,
    damping=2.5,
    growth_rate=0.015,
    spring_length_thresholds: float | list[float] = [
        20.0,
        20.0,
        8.0,
        20.0,
        20.0,
        20.0,
    ],
    growth_rate_distrs=None,
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
        raise TypeError("")

    if growth_rate_distrs is None:
        growth_rate_distrs = [(0.0, 0.0)] * len(initial_positions)
    agents = [
        crm.RodAgent(
            pos,
            vel=0 * pos,
            interaction=interaction,
            diffusion_constant=diffusion_constant,
            spring_tension=spring_tension,
            rigidity=rigidity,
            spring_length=spring_length,
            damping=damping,
            growth_rate=growth_rate,
            growth_rate_distr=growth_rate_distr,
            spring_length_threshold=spring_length_threshold,
            neighbor_reduction=None,
        )
        for spring_length_threshold, pos, growth_rate_distr in zip(
            spring_length_thresholds, initial_positions, growth_rate_distrs
        )
    ]

    # define config
    config = settings.to_config()
    config.show_progressbar = True

    container = crm.run_simulation_with_agents(config, agents)
    print()
    return container


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


def main():
    masks_data, positions_initial, settings, iterations_all, mask_iters = (
        preprocessing()
    )

    container = predict(
        positions_initial, settings, np.max(iterations_all) - np.min(iterations_all)
    )
    iterations = np.array(container.get_all_iterations()).astype(int)

    adjust_masks(masks_data, mask_iters, container, settings)

    exit()

    masks_predicted = [
        crm.render_mask(
            container.get_cells_at_iteration(iter),
            container.cell_to_color,
            settings.constants.domain_size,
            render_settings=crm.RenderSettings(pixel_per_micron=1),
        )
        for iter in tqdm(iterations, total=len(iterations), desc="Render Masks")
    ]

    masks_colors = [
        crm_fit.main.transform_input_mask(
            colors_data, mask_data, iteration, container, return_colors=True
        )
        for colors_data, mask_data, iteration in tqdm(
            zip(colors_all, masks, iterations),
            total=len(masks),
            desc="Transform Input Masks",
        )
    ]
    masks_input = [m[0] for m in masks_colors]
    color_mappings = [m[1] for m in masks_colors]

    color_to_cell = {}
    parent_map = container.parent_map
    for i, cm in enumerate(color_mappings):
        for n, c in enumerate(cm.values()):
            if i > 0:
                id = crm.CellIdentifier(crm.VoxelPlainIndex(0), n)
            else:
                id = crm.CellIdentifier.new_initial(n)
            if c not in color_to_cell.keys():
                color_to_cell[c] = id

    print("## Color -> Cell")
    for k in color_to_cell.keys():
        included = color_to_cell[k] in parent_map.keys()
        print(included, k, "-->", color_to_cell[k])

    print("## Cell -> Parent")
    for k in parent_map.keys():
        included = k in color_to_cell.values()
        print(included, k, "-->", parent_map[k])

    fig, axs = plt.subplots(1, 2)

    for i, m in enumerate([masks[0], masks[-1]]):
        colors = np.unique(m)
        axs[i].imshow(m)

        for c in colors:
            middle = np.mean(np.where(m == c), axis=1)
            print(middle)
            axs[i].text(middle[1], middle[0], str(c), color="white")
    plt.show()

    # WARNING!
    # THIS PART HERE IS VERY MANUAL AND SHOULD
    # BE REPLACED IF THIS SHOULD BE ADAPTED
    # INTO SOMETHING THAT IS MORE AUTOMATED!
    # parent_map[crm.CellIdentifier.new(1, 6)] = crm.CellIdentifier.new_initial(5)
    # parent_map[crm.CellIdentifier.new(1, 7)] = crm.CellIdentifier.new_initial(5)

    diffs = [
        crm.penalty_area_diff(m1, m2) for m1, m2 in zip(masks_input, masks_predicted)
    ]
    diffs_parents = [
        crm.penalty_area_diff_account_parents(m1, m2, color_to_cell, parent_map, 0)
        for m1, m2 in zip(masks_input, masks_predicted)
    ]

    _, ax = plt.subplots()
    crm.configure_ax(ax)
    ax.plot(diffs)
    plt.show()
