import cv2 as cv
import cr_mech_coli as crm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from glob import glob
import time
from pathlib import Path
import multiprocessing as mp
import argparse
import scipy as sp

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


def exponential_growth(t, grate, x0):
    return x0 * np.exp(grate * t)


def estimate_growth_rates(iterations, lengths, settings, out_path):
    times = np.array(iterations) / (len(iterations) - 1) * settings.constants.t_max
    popts = []
    pcovs = []
    growth_rates = []
    growth_rates_err = []
    for i in range(len(lengths[0])):
        x0 = lengths[0][i]
        xf = lengths[-1][i]
        popt, pcov = sp.optimize.curve_fit(
            exponential_growth,
            times,
            [length[i] for length in lengths],
            p0=(np.log(xf / x0) / settings.constants.t_max, x0),
        )
        popts.append(popt)
        pcovs.append(pcov)
        growth_rates.append(popt[0])
        growth_rates_err.append(np.sqrt(pcov[0, 0]))

    growth_rates = np.array(growth_rates)
    growth_rates_err = np.array(growth_rates_err)

    fig, ax = plt.subplots(figsize=(8, 8))
    crm.plotting.configure_ax(ax)
    ax.plot(times, lengths, color=crm.plotting.COLOR5)
    for n, popt, pcov in zip(range(len(growth_rates)), popts, pcovs):
        ax.plot(
            times,
            exponential_growth(times, *popt),
            color=crm.plotting.COLOR3,
        )
        ax.fill_between(
            times,
            exponential_growth(times, *(popt - np.array([pcov[0, 0] ** 0.5, 0]))),
            exponential_growth(times, *(popt + np.array([pcov[0, 0] ** 0.5, 0]))),
            color=crm.plotting.COLOR1,
            alpha=0.3,
        )

    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Rod Length [pix]")

    fig.savefig(out_path / "estimated-growth-rates.png")
    fig.savefig(out_path / "estimated-growth-rates.pdf")
    return growth_rates, growth_rates_err


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

    imgs = [cv.imread(fi) for fi in files_images]

    masks = [np.loadtxt(fm, delimiter=",") for fm in files_masks]

    print(f"{time.time() - interval:10.4f}s Loaded data")
    interval = time.time()

    n_vertices = settings.constants.n_vertices
    domain_size = settings.constants.domain_size

    iterations = []
    positions = []
    lengths = []
    for mask, filename in zip(masks, files_masks):
        try:
            pos, length, _ = crm.extract_positions(
                mask, n_vertices, domain_size=domain_size
            )
            positions.append(pos)
            lengths.append(length)
            iterations.append(int(Path(filename).stem.split("-")[0]))
        except ValueError as e:
            print("Encountered Error during extraction of positions:")
            print(filename)
            print(e)
            print("Omitting this particular result.")

    iterations = np.array(iterations) - iterations[0]
    growth_rates, _ = estimate_growth_rates(iterations, lengths, settings, out)
    settings.constants.n_saves = max(iterations)

    settings.parameters.growth_rate = list(growth_rates)

    print(f"{time.time() - interval:10.4f}s Calculated initial values")
    interval = time.time()

    n_agents = positions[0].shape[0]
    lower, upper, x0, param_infos, constants, constant_infos = (
        settings.generate_optimization_infos(n_agents)
    )
    bounds = np.array([lower, upper]).T

    # Fix some parameters for the simulation
    args_predict = (
        iterations,
        positions,
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
        positions[0],
        settings,
    )

    if cell_container is None:
        print("Best fit does not produce valid result.")
        exit()

    iterations = cell_container.get_all_iterations()
    agents_predicted = cell_container.get_cells_at_iteration(iterations[-1])

    if not pyargs.skip_masks:

        def plot_snapshot(pos, img, name):
            for p in pos:
                p = crm.convert_cell_pos_to_pixels(p, domain_size, img.shape[:2])
                img = cv.polylines(
                    np.array(img),
                    [np.round(p).astype(int)],
                    isClosed=False,
                    color=(250, 250, 250),
                    thickness=1,
                )
            cv.imwrite(f"{out}/{name}.png", img)

        for iter, img in zip(iterations, imgs):
            agents = cell_container.get_cells_at_iteration(iter)
            pos = np.array([c[0].pos for c in agents.values()])
            plot_snapshot(pos, img, f"snapshot-{iter:06}")

        print(f"{time.time() - interval:10.4f}s Rendered Masks")
        interval = time.time()

    if not pyargs.skip_param_space:
        visualize_param_space(out, param_infos, final_params, final_cost)
        print(f"{time.time() - interval:10.4f}s Visualized parameter space")
        interval = time.time()

    if not pyargs.skip_distributions:
        plot_distributions(agents_predicted, out)
        print(f"{time.time() - interval:10.4f}s Plotted Distributions")
