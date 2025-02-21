import cr_mech_coli as crm
import numpy as np
import cv2 as cv
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import argparse

mpl.use("pgf")
plt.rcParams.update(
    {
        "font.family": "serif",  # use serif/main font for text elements
        "text.usetex": True,  # use inline math for ticks
        "pgf.rcfonts": False,  # don't setup fonts from rc parameters
        "pgf.preamble": "\\usepackage{siunitx}",  # load additional packages
    }
)


def calculate_lengths_distances(
    n, ccs, domain_size, skel_method, n_vertices
) -> tuple[list, list, list]:
    cell_container = crm.CellContainer.deserialize(ccs)
    cells_at_iteration = cell_container.get_cells()[n]
    colors = cell_container.cell_to_color

    mask = crm.render_mask(cells_at_iteration, colors, domain_size)
    positions = np.array(
        crm.extract_positions(mask, n_vertices=n_vertices, skel_method=skel_method)[0]
    )

    distances = []
    lengths_extracted = []
    lengths_exact = []
    for p in positions:
        color = mask[int(p[0][1]), int(p[0][0])]
        ident = cell_container.get_cell_from_color([*color])
        cell = cells_at_iteration[ident][0]
        q = cell.pos[:, :2]
        p = crm.convert_pixel_to_position(p, config.domain_size, mask.shape[:2])

        # Determine if we need to use the reverse order
        d1t = np.sum((p - q) ** 2, axis=1) ** 0.5
        d2t = np.sum((p - q[::-1]) ** 2, axis=1) ** 0.5
        d1 = np.sum(d1t)
        d2 = np.sum(d2t)
        # d = min(d1, d2) / len(p)
        if d1 <= d2:
            distances.append(d1t / len(p))
        else:
            distances.append(d2t / len(p))
        # distances_i.append(d)

        # Compare total length
        l1 = np.sum((p[1:] - p[:-1]) ** 2) ** 0.5
        l2 = np.sum((q[1:] - q[:-1]) ** 2) ** 0.5
        lengths_extracted.append(l1)
        lengths_exact.append(l2)
    return distances, lengths_extracted, lengths_exact


def create_simulation_result(n_vertices: int, rng_seed: int = 1):
    n_agents = 4
    config = crm.Configuration(
        t0=0.0,
        dt=0.02,
        t_max=200.0,
        save_interval=4.0,
        domain_size=100,
    )
    agent_settings = crm.AgentSettings(growth_rate=0.05)
    agent_settings.mechanics.rigidity = 2.0
    config.domain_height = 0.2

    positions = crm.generate_positions_old(
        n_agents,
        agent_settings,
        config,
        rng_seed=rng_seed,
        dx=config.domain_size / 10.0,
        randomize_positions=0.0,
        n_vertices=n_vertices,
    )
    rod_args = agent_settings.to_rod_agent_dict()
    agents = [crm.RodAgent(pos=p, vel=p * 0.0, **rod_args) for p in positions]
    return config, crm.run_simulation_with_agents(config, agents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot comparison between exact and extracted positions",
    )
    parser.add_argument(
        "-n",
        "--n-vertices",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--skel-method",
        default="lee",
        help='Skeletonization method. Can be "lee" or "zhang"',
    )
    parser.add_argument("-w", "--workers", type=int, default=-1)
    parser.add_argument("--skip-masks", action="store_true", default=False)
    parser.add_argument("--skip-graph", action="store_true", default=False)
    parser.add_argument("--skip-distribution", action="store_true", default=False)
    pyargs = parser.parse_args()

    config, cell_container = create_simulation_result(pyargs.n_vertices)
    all_cells = cell_container.get_cells()
    iterations = cell_container.get_all_iterations()
    colors = cell_container.cell_to_color

    if not pyargs.skip_masks:
        # Pick one iteration to plot results
        indices = [9, 19, 29, 39, 49]
        iter_masks = [
            (
                iterations[ind],
                crm.render_mask(all_cells[iterations[ind]], colors, config.domain_size),
            )
            for ind in indices
        ]
        for iteration, mask in tqdm(iter_masks):
            positions = crm.extract_positions(
                mask, n_vertices=pyargs.n_vertices, skel_method=pyargs.skel_method
            )[0]
            positions = np.round(np.array(positions))
            positions = np.array(positions, dtype=int).reshape(
                (len(positions), -1, 1, 2)
            )

            dl = 2**0.5 * config.domain_size
            domain_pixels = np.array(mask.shape[:2], dtype=float)
            pixel_per_length = domain_pixels / dl

            # Calculate differences in positions
            pos_exact = []
            for p1 in positions:
                # Get color
                color = mask[p1[0][0][1], p1[0][0][0]]
                ident = cell_container.get_cell_from_color([*color])
                cell = all_cells[iteration][ident][0]
                p1 = np.array(p1[:, 0, :], dtype=float)
                p2 = crm.convert_cell_pos_to_pixels(
                    cell.pos, config.domain_size, mask.shape[:2]
                )
                pos_exact.append(p2)

            pos_exact = np.round(np.array(pos_exact)).reshape(
                (len(pos_exact), -1, 1, 2)
            )
            pos_exact = np.array(pos_exact, dtype=int)
            # mask = cv.polylines(mask, positions, False, (50, 50, 50), 2)
            mask = cv.polylines(mask, pos_exact, False, (150, 150, 150), 2)
            mask = cv.polylines(mask, pos_exact, False, (250, 250, 250), 1)
            for p in positions.reshape((-1, 2)):
                mask = cv.drawMarker(
                    mask, p, (50, 50, 50), cv.MARKER_TILTED_CROSS, 14, 2
                )
            path = Path("docs/source/_static/fitting-methods/")
            cv.imwrite(
                filename=str(path / "extract_positions-{:06}.png".format(iteration)),
                img=mask,  # [200:-200, 200:-200],
            )

    ccs = cell_container.serialize()
    arglist = [
        (n, ccs, config.domain_size, pyargs.skel_method, pyargs.n_vertices)
        for n in iterations
    ]

    if not pyargs.skip_graph or not pyargs.skip_distribution:
        if pyargs.workers < 0:
            import multiprocessing as mp

            pool = mp.Pool()
            results = pool.starmap(calculate_lengths_distances, arglist)
        elif pyargs.workers == 1:
            results = [calculate_lengths_distances(*a) for a in arglist]
        else:
            import multiprocessing as mp

            pool = mp.Pool(pyargs.workers)
            results = pool.starmap(calculate_lengths_distances, arglist)
        distances = [np.sum(r[0]) / pyargs.n_vertices for r in results]
        distances_vertices = [np.array(r[0]).reshape(-1) for r in results]
        lengths_extracted = [r[1] for r in results]
        lengths_exact = [r[2] for r in results]
    else:
        exit()

    if not pyargs.skip_distribution:
        fig, ax = plt.subplots()

        t = 0.7
        q = len(distances_vertices)
        colors = [(t * i / q, t * i / q, t * i / q) for i in range(q)]
        ax.set_title("Distribution of distances between individual vertices")
        n_bins = 50
        logbins = np.logspace(
            np.log10(np.min([np.min(d) for d in distances_vertices])),
            np.log10(np.max([np.max(d) for d in distances_vertices])),
            n_bins + 1,
        )
        ax.hist(
            distances_vertices,
            logbins,
            stacked=True,
            color=colors,
            label="Distance between vertices",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Distance [$\\SI{}{\\micro\\metre}$]")
        ax.set_ylabel("Count")
        ax.legend()
        fig.savefig("docs/source/_static/fitting-methods/displacement-distribution.png")
        plt.close(fig)

    if not pyargs.skip_graph:
        fig, ax1 = plt.subplots()

        x = np.arange(len(distances)) * config.save_interval
        ax1.plot(
            x,
            [np.mean(li) for li in lengths_extracted],
            # yerr=[np.std(li) for li in lengths1],
            linestyle="--",
            color="k",
            label="Avg. Rod Length (Extracted)",
        )
        ax1.plot(
            x,
            [np.mean(li) for li in lengths_exact],
            # yerr=[np.std(li) for li in lengths2],
            linestyle=":",
            color="k",
            label="Avg. Rod Length (Exact)",
        )
        ax1.fill_between(
            x,
            y1=[
                np.mean(lengths_extracted[i]) - np.mean(distances[i])
                for i in range(len(lengths_extracted))
            ],
            y2=[
                np.mean(lengths_extracted[i]) + np.mean(distances[i])
                for i in range(len(lengths_extracted))
            ],
            alpha=0.3,
            color="gray",
            label="Avg. Vertex Distance",
        )
        ax1.legend()
        ax1.set_ylabel("Length [µm]")
        ax1.set_xlabel("Time [min]")
        ax1.set_title("Evaluation of Position Extraction Algorithm")
        fig.savefig("docs/source/_static/fitting-methods/displacement-calculations.png")
        plt.close(fig)
