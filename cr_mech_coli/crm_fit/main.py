import cv2 as cv
import cr_mech_coli as crm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from glob import glob
import scipy as sp
import time
from pathlib import Path
import multiprocessing as mp
import argparse

from .plotting import plot_profile, plot_distributions, visualize_param_space
from .predict import predict_flatten, predict, store_parameters


# Create folder to store output
def get_out_folder(iteration: int | None, potential_type) -> Path:
    base = Path(f"./out/crm_fit/{potential_type.to_short_string()}")
    if iteration is not None:
        out = base / f"{iteration:04}"
    else:
        folders = sorted(glob(str(base / "*")))
        if len(folders) > 0:
            n = int(folders[-1].split("/")[-1]) + 1
        else:
            n = 0
        out = base / f"{n:04}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def crm_fit_main():
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
        "-w", "--workers", type=int, default=-1, help="Number of threads"
    )
    parser.add_argument(
        "data",
        help="Directory containing initial and final snapshots with masks.",
    )
    parser.add_argument(
        "-o",
        "--output-folder",
        help="Folder to store all output in. If left unspecified, the output folder will be\
            generated via OUTPUT_FOLDER='./out/crm_fit/POTENTIAL_TYPE/ITERATION/' where ITERATION\
            is the next number larger than any already existing one and POTENTIAL_TYPE is obtained\
            from the settings.toml file",
    )
    parser.add_argument(
        "--skip-profiles",
        default=False,
        help="Skips Plotting of profiles for parameters",
        action="store_true",
    )
    parser.add_argument(
        "--skip-masks",
        default=False,
        help="Skips Plotting of masks and microscopic images",
        action="store_true",
    )
    parser.add_argument(
        "--skip-param-space",
        default=False,
        help="Skips visualization of parameter space",
        action="store_true",
    )
    parser.add_argument(
        "--skip-distributions",
        default=False,
        help="Skips plotting of distributions",
        action="store_true",
    )
    pyargs = parser.parse_args()
    if pyargs.workers == -1:
        pyargs.workers = mp.cpu_count()

    if pyargs.data is None:
        dirs = sorted(glob("data/*"))
        if len(dirs) == 0:
            raise ValueError('Could not find any directory inside "./data/"')
        data_dir = Path(dirs[0])
    else:
        data_dir = Path(pyargs.data)
    files_images = sorted(glob(str(data_dir / "images/*")))
    files_masks = sorted(glob(str(data_dir / "masks/*.csv")))

    mpl.use("pgf")
    plt.rcParams.update(
        {
            "font.family": "serif",  # use serif/main font for text elements
            "text.usetex": True,  # use inline math for ticks
            "pgf.rcfonts": False,  # don't setup fonts from rc parameters
            "pgf.preamble": "\\usepackage{siunitx}",  # load additional packages
        }
    )

    # Try to read config file
    filename = data_dir / "settings.toml"
    settings = crm.crm_fit.Settings.from_toml(str(filename))
    potential_type = settings.parameters.potential_type

    out = get_out_folder(pyargs.iteration, potential_type)
    if pyargs.output_folder is not None:
        out = Path(pyargs.output_folder)

    interval = time.time()

    img1 = cv.imread(files_images[0])
    img2 = cv.imread(files_images[1])

    mask1 = np.loadtxt(files_masks[0], delimiter=",").T
    mask2 = np.loadtxt(files_masks[1], delimiter=",").T

    print(f"{time.time() - interval:10.4f}s Loaded data")
    interval = time.time()

    n_vertices = settings.constants.n_vertices
    domain_size = settings.constants.domain_size
    pos1 = crm.extract_positions(mask1, n_vertices, domain_size=domain_size)[0]
    pos2 = crm.extract_positions(mask2, n_vertices, domain_size=domain_size)[0]

    print(f"{time.time() - interval:10.4f}s Calculated initial values")
    interval = time.time()

    n_agents = pos1.shape[0]
    lower, upper, x0, param_infos, constants, constant_infos = (
        settings.generate_optimization_infos(n_agents)
    )
    bounds = np.array([lower, upper]).T

    # Fix some parameters for the simulation
    args_predict = (
        pos1,
        pos2,
        settings,
        out,
    )

    filename = "final_params.csv"
    if (out / filename).exists():
        params = np.genfromtxt(out / filename, delimiter=",")
        final_params = params[:-1]
        final_cost = params[-1]
        print(f"{time.time() - interval:10.4f}s Found previous results")
    else:
        res = sp.optimize.differential_evolution(
            predict_flatten,
            bounds=bounds,
            x0=x0,
            args=args_predict,
            workers=pyargs.workers,
            updating="deferred",
            maxiter=settings.optimization.max_iter,
            # constraints=constraints,
            disp=True,
            tol=settings.optimization.tol,
            recombination=settings.optimization.recombination,
            popsize=settings.optimization.pop_size,
            polish=False,
            rng=settings.optimization.seed,
        )
        final_cost = res.fun
        final_params = res.x
        # Store information in file
        store_parameters(final_params, filename, out, final_cost)
        print(f"{time.time() - interval:10.4f}s Finished Parameter Optimization")

    interval = time.time()

    # Plot Cost function against varying parameters
    if not pyargs.skip_profiles:
        for n, (p, bound) in enumerate(zip(final_params, bounds)):
            fig_ax = None
            fig_ax = plot_profile(
                n,
                bound,
                args_predict[:-1],
                param_infos[n],
                final_params,
                final_cost,
                out,
                pyargs.workers,
                fig_ax,
                steps=40,
            )
            fig, _ = fig_ax
            plt.close(fig)

        print(f"{time.time() - interval:10.4f} Plotted Profiles")
        interval = time.time()

    cell_container = predict(
        final_params,
        pos1,
        settings,
    )

    if cell_container is None:
        print("Best fit does not produce valid result.")
        exit()

    iterations = cell_container.get_all_iterations()
    agents_predicted = cell_container.get_cells_at_iteration(iterations[-1])

    if not pyargs.skip_masks:
        figs_axs = [plt.subplots() for _ in range(4)]

        def plot_snapshot(pos, img, name):
            for p in pos:
                p = crm.convert_cell_pos_to_pixels(p, domain_size, img1.shape[:2][::-1])
                img = cv.polylines(
                    np.array(img),
                    [np.round(p).astype(int)],
                    isClosed=False,
                    color=(250, 250, 250),
                    thickness=1,
                )
            cv.imwrite(f"{out}/{name}.png", img)

        plot_snapshot(pos1, img1, "snapshot-initial")
        plot_snapshot(pos2, img2, "snapshot-final")

        pos_final = np.array([c[0].pos for c in agents_predicted.values()])
        plot_snapshot(pos_final, img2, "snapshot-final-predicted")

        print(f"{time.time() - interval:10.4f}s Rendered Masks")
        interval = time.time()

    if not pyargs.skip_param_space:
        visualize_param_space(out, param_infos, final_params, final_cost)
        print(f"{time.time() - interval:10.4f}s Visualized parameter space")
        interval = time.time()

    if not pyargs.skip_distributions:
        plot_distributions(agents_predicted, out)
        print(f"{time.time() - interval:10.4f}s Plotted Distributions")
