import cr_mech_coli as crm
import numpy as np
import cv2 as cv
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import argparse
import time
from PIL import Image
import scipy as sp
import multiprocessing as mp

OPATH = Path("docs/source/_static/fitting-methods")


def calculate_lengths_distances(
    n, ccs, domain_size, skel_method, n_vertices
) -> tuple[list, list, list, list]:
    cell_container = crm.CellContainer.deserialize(ccs)
    cells_at_iteration = cell_container.get_cells()[n]
    colors = cell_container.cell_to_color

    mask = crm.render_mask(cells_at_iteration, colors, domain_size)
    positions = np.array(
        crm.extract_positions(
            mask,
            n_vertices=n_vertices,
            skel_method=skel_method,
        )[0]
    )

    directed_diffs = []
    distances = []
    lengths_extracted = []
    lengths_exact = []
    for p in positions:
        color = mask[int(np.round(p[0][0])), int(np.round(p[0][1]))]
        ident = cell_container.get_cell_from_color((color[0], color[1], color[2]))
        cell = cells_at_iteration[ident][0]
        q = cell.pos[:, :2]
        p = crm.convert_pixel_to_position(p, config.domain_size, mask.shape[:2])

        # Determine if we need to use the reverse order
        d1t = np.sum((p - q) ** 2, axis=1) ** 0.5
        d2t = np.sum((p - q[::-1]) ** 2, axis=1) ** 0.5
        d1 = np.sum(d1t)
        d2 = np.sum(d2t)

        dirs_exact = np.zeros(p.shape)
        dirs_exact[0] = (p[1] - p[0]) / np.linalg.norm(p[1] - p[0])
        dirs_exact[-1] = (p[-1] - p[-2]) / np.linalg.norm(p[-1] - p[-2])
        norm = np.linalg.norm(p[2:] - p[0:-2], axis=1)
        dirs_exact[1:-1] = (p[2:] - p[0:-2]) / norm[:, np.newaxis]
        rotation_mats = np.array([[[d[0], d[1]], [-d[1], d[0]]] for d in dirs_exact])
        if d1 <= d2:
            distances.append(d1t / len(p))
            diff = q - p
        else:
            distances.append(d2t / len(p))
            diff = q[::-1] - p

        rotated = np.array([r @ v for r, v in zip(rotation_mats, diff)])
        directed_diffs.append(rotated)

        # Compare total length
        l1 = np.sum((p[1:] - p[:-1]) ** 2) ** 0.5
        l2 = np.sum((q[1:] - q[:-1]) ** 2) ** 0.5
        lengths_extracted.append(l1)
        lengths_exact.append(l2)
    return directed_diffs, distances, lengths_extracted, lengths_exact


def calculate_lengths_distances_wrapper(args):
    return calculate_lengths_distances(*args)


def create_simulation_result(n_vertices: int, rng_seed: int = 3):
    interval = time.time()
    n_agents = 4
    config = crm.Configuration(
        t0=0.0,
        dt=0.02,
        t_max=200.0,
        n_saves=49,
        domain_size=np.array([200, 200]),
    )
    config.storage_options = [crm.StorageOption.Memory]
    config.progressbar = ""
    agent_settings = crm.AgentSettings(
        growth_rate=0.012,
        growth_rate_setter={"mean": 0.012, "std": 0.002},
    )
    agent_settings.mechanics.rigidity = 8.0
    config.domain_height = 0.2

    positions = crm.generate_positions(
        n_agents,
        agent_settings,
        config,
        rng_seed=rng_seed,
        dx=np.array(config.domain_size) * 0.2,
        randomize_positions=0.0,
        n_vertices=n_vertices,
    )
    rod_args = agent_settings.to_rod_agent_dict()
    agents = [crm.RodAgent(pos=p, vel=p * 0.0, **rod_args) for p in positions]
    rng = np.random.default_rng(rng_seed)
    for a in agents:
        a.growth_rate += 0.002 * rng.random(1)
    res = crm.run_simulation_with_agents(config, agents)
    print(f"{time.time() - interval:8.4} Created Simulation Result:")
    return config, res


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
            # This generates extracted positions in pixel units
            positions = crm.extract_positions(
                mask,
                n_vertices=pyargs.n_vertices,
                skel_method=pyargs.skel_method,
                # domain_size=config.domain_size,
            )[0]
            positions = np.array(np.round(positions), dtype=int)

            # Calculate differences in positions
            pos_exact = []
            for n, p0 in enumerate(positions):
                # Get color
                color = mask[p0[0][0], p0[0][1]]
                ident = cell_container.get_cell_from_color(
                    (color[0], color[1], color[2])
                )
                cell = all_cells[iteration][ident][0]
                p1 = crm.convert_cell_pos_to_pixels(
                    cell.pos[:, :2], config.domain_size, mask.shape[:2]
                )
                p2 = crm.convert_pixel_to_position(
                    p0, config.domain_size, mask.shape[:2]
                )
                mask = cv.polylines(
                    mask,
                    [p0[:, ::-1]],
                    isClosed=False,
                    color=(250, 250, 250),
                    thickness=1,
                )
                for pi in p1[:, ::-1]:
                    mask = cv.drawMarker(
                        mask,
                        pi,
                        (50, 50, 50),
                        cv.MARKER_TILTED_CROSS,
                        14,
                        2,
                    )

            cv.imwrite(
                filename=str(OPATH / f"extract_positions-{iteration:06}.png"),
                img=mask,
            )
            pil_img = Image.fromarray(mask)
            pil_img.save(str(OPATH / f"extract_positions-{iteration:06}.pdf"))

    ccs = cell_container.serialize()
    arglist = [
        (n, ccs, config.domain_size, pyargs.skel_method, pyargs.n_vertices)
        for n in iterations
    ]

    if not pyargs.skip_graph or not pyargs.skip_distribution:
        crm.plotting.set_mpl_rc_params()
        try:
            directed_diffs = np.load(OPATH / "directed_diffs.npy", allow_pickle=True)
            distances = np.load(OPATH / "distances.npy", allow_pickle=True)
            distances_vertices = np.load(
                OPATH / "distances_vertices.npy", allow_pickle=True
            )
            lengths_extracted = np.load(
                OPATH / "lengths_extracted.npy", allow_pickle=True
            )
            lengths_exact = np.load(OPATH / "lengths_exact.npy", allow_pickle=True)
        except:
            if pyargs.workers < 0:
                import multiprocessing as mp

                pool = mp.Pool()
                results = list(
                    tqdm(
                        pool.imap(calculate_lengths_distances_wrapper, arglist),
                        total=len(arglist),
                    ),
                )
            elif pyargs.workers == 1:
                results = [
                    calculate_lengths_distances(*a)
                    for a in tqdm(arglist, total=len(arglist))
                ]
            else:
                pool = mp.Pool(pyargs.workers)
                results = list(
                    tqdm(
                        pool.imap(calculate_lengths_distances_wrapper, arglist),
                        total=len(iterations),
                    )
                )
            directed_diffs = [r[0] for r in results]
            distances = [np.sum(r[1]) / pyargs.n_vertices for r in results]
            distances_vertices = [np.array(r[1]).reshape(-1) for r in results]
            lengths_extracted = [r[2] for r in results]
            lengths_exact = [r[3] for r in results]

            # Store results in files
            def store_list_of_arrays(name, li):
                OPATH.mkdir(parents=True, exist_ok=True)
                np.save(OPATH / name, np.array(li, dtype=object))

            store_list_of_arrays("directed_diffs", directed_diffs)
            store_list_of_arrays("distances", distances)
            store_list_of_arrays("distances_vertices", distances_vertices)
            store_list_of_arrays("lengths_extracted", lengths_extracted)
            store_list_of_arrays("lengths_exact", lengths_exact)
    else:
        exit()

    if not pyargs.skip_distribution:
        fig, ax = plt.subplots(figsize=(8, 8))
        crm.plotting.configure_ax(ax)

        all_points = np.vstack(directed_diffs).reshape((-1, 2))
        c = sp.stats.gaussian_kde(all_points.T)(all_points.T)
        ax.scatter(
            all_points[:, 0],
            all_points[:, 1],
            c=c,
            cmap=crm.plotting.cmap,
            marker=".",
        )
        ax.set_title("Vertex Displacement (3σ)")
        ax.set_xlabel("Parallel to Segment [µm]")
        ax.set_ylabel("Orthogonal to Segment [µm]")

        dx = np.percentile(np.abs(all_points), 99.73)
        # dx = np.max(np.abs(all_points))
        ax.set_xlim(-1.2 * dx, 1.2 * dx)
        ax.set_ylim(-1.2 * dx, 1.2 * dx)

        fig.savefig(OPATH / "displacement-distribution.png")
        fig.savefig(OPATH / "displacement-distribution.pdf")
        plt.close(fig)

        c1 = np.array(mpl.colors.to_rgba(crm.plotting.COLOR3))
        c2 = np.array(mpl.colors.to_rgba(crm.plotting.COLOR1))
        q = len(directed_diffs)
        colors = [c2 * i / q + (1 - i / q) * c1 for i in range(q)]

        for i, name in enumerate(["x", "y"]):
            fig, ax = plt.subplots(figsize=(8, 8))
            crm.plotting.configure_ax(ax)

            ax.hist(
                [np.array(di)[:, :, i].reshape(-1) for di in directed_diffs],
                bins=100,
                stacked=True,
                color=colors,
                label="Data",
            )

            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.10),
                ncol=1,
                frameon=False,
            )
            ax.set_yscale("log")
            ax.set_xlim(-1.1 * dx, 1.1 * dx)

            fig.savefig(OPATH / f"displacement-distr-{name}.png")
            fig.savefig(OPATH / f"displacement-distr-{name}.pdf")
            plt.close(fig)

        # Now do plot over time
        def gauss2d(x, y, mux, muy, sigmax, sigmay, prefactor):
            gx = sp.stats.norm.pdf(x, mux, sigmax)
            gy = sp.stats.norm.pdf(y, muy, sigmay)
            return prefactor * gx * gy

        means = []
        covs = []
        for data in directed_diffs:
            data = np.array(data).reshape((-1, 2))
            filt = np.all(np.abs(data) <= dx, axis=1)
            data = data[filt]
            mean, cov = sp.stats.multivariate_normal.fit(data)
            means.append(mean)
            covs.append(cov)

        means = np.array(means)
        covs = np.array(covs)

        fig, ax = plt.subplots(figsize=(8, 8))
        crm.plotting.configure_ax(ax)

        t = np.arange(len(directed_diffs)) * config.t_max / (config.n_saves + 1)
        ax.plot(
            t,
            means[:, 0],
            color=crm.plotting.COLOR5,
            label="parallel",
            linestyle="-",
        )
        ax.fill_between(
            t,
            means[:, 0] - covs[:, 0, 0] ** 0.5,
            means[:, 0] + covs[:, 0, 0] ** 0.5,
            color=crm.plotting.COLOR5,
            alpha=0.5,
        )
        ax.plot(
            t,
            means[:, 1],
            color=crm.plotting.COLOR3,
            label="orthogonal",
            linestyle="--",
        )
        ax.fill_between(
            t,
            means[:, 1] - covs[:, 1, 1] ** 0.5,
            means[:, 1] + covs[:, 1, 1] ** 0.5,
            color=crm.plotting.COLOR3,
            alpha=0.5,
        )
        dmean = np.max(np.abs(means))
        dcovs = np.max([np.abs(covs[:, 0, 0]) ** 0.5, np.abs(covs[:, 1, 1]) ** 0.5])
        dlim = np.max([1.2 * dmean, dmean + dcovs])
        ax.set_ylim(-dlim, dlim)
        ax.set_xlabel("Time [min]")
        ax.set_ylabel("Mean [µm]")
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.10),
            ncol=2,
            frameon=False,
        )
        fig.savefig(OPATH / "displacement-fit-over-time.png")
        fig.savefig(OPATH / "displacement-fit-over-time.pdf")
        plt.close(fig)

    if not pyargs.skip_graph:
        fig, ax = plt.subplots(figsize=(8, 8))
        crm.plotting.configure_ax(ax)

        x = np.arange(len(distances)) * config.t_max / (config.n_saves + 1)
        ax.plot(
            x,
            [np.mean(li) for li in lengths_exact],
            # yerr=[np.std(li) for li in lengths2],
            linestyle="-",
            color=crm.plotting.COLOR5,
            label="Exact",
        )
        ax.plot(
            x,
            [np.mean(li) for li in lengths_extracted],
            # yerr=[np.std(li) for li in lengths1],
            linestyle=":",
            color=crm.plotting.COLOR3,
            label="Extracted",
        )
        ax.fill_between(
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
            color=crm.plotting.COLOR1,
        )
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.10),
            ncol=3,
            frameon=False,
        )
        ax.set_ylabel("Rod Length [µm]")
        ax.set_xlabel("Time [min]")
        # ax.set_title("Evaluation of Position Extraction Algorithm")
        fig.savefig(OPATH / "displacement-calculations.png")
        fig.savefig(OPATH / "displacement-calculations.pdf")
        plt.close(fig)
