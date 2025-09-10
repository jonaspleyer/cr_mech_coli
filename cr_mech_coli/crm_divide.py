import numpy as np
from glob import glob
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

import cr_mech_coli as crm
from cr_mech_coli import crm_fit

data_dir = Path("data/crm_divide/0001/")

crm.plotting.set_mpl_rc_params()


def adjust_masks(masks, mask_iters, container: crm.CellContainer, settings):
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
    for m in tqdm(masks, total=len(masks), desc="Adjusting Data Masks"):
        new_mask = np.array(
            [color_data_transform[c] for c in m.reshape(-1)], dtype=np.uint8
        ).reshape((*m.shape, 3))
        new_masks.append(new_mask)

    color_to_cell = {v: k for k, v in cell_to_color.items()}

    return new_masks, parent_map, cell_to_color, color_to_cell


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
    growth_rate=0.02,
    spring_length_thresholds: float | list[float] = [
        200.0,
        200.0,
        8.0,
        11.0,
        200.0,
        200.0,
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


def plot_time_evolution(
    masks_predicted,
    new_masks,
    color_to_cell,
    parent_map,
    iterations_simulation,
    iterations_data,
    settings,
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
    return fig


def main():
    masks_data, positions_initial, settings, iterations_data, mask_iters = (
        preprocessing()
    )

    container = predict(
        positions_initial, settings, np.max(iterations_data) - np.min(iterations_data)
    )
    iterations_simulation = np.array(container.get_all_iterations()).astype(int)

    new_masks, parent_map, cell_to_color, color_to_cell = adjust_masks(
        masks_data, mask_iters, container, settings
    )

    masks_predicted = [
        crm.render_mask(
            container.get_cells_at_iteration(iter),
            cell_to_color,
            settings.constants.domain_size,
            render_settings=crm.RenderSettings(pixel_per_micron=1),
        )
        for iter in tqdm(
            iterations_simulation, total=len(iterations_simulation), desc="Render Masks"
        )
    ]

    fig = plot_time_evolution(
        masks_predicted,
        new_masks,
        color_to_cell,
        parent_map,
        iterations_simulation,
        iterations_data,
        settings,
    )
    plt.show()
    plt.close(fig)
