from cv2 import imread
import cr_mech_coli as crm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use("pgf")

import scipy as sp
import time
from pathlib import Path
import multiprocessing as mp
import argparse

from plotting import plot_profile, plot_distributions, visualize_param_space
from predict import predict_flatten, predict, store_parameters, PotentialType

plt.rcParams.update(
    {
        "font.family": "serif",  # use serif/main font for text elements
        "text.usetex": True,  # use inline math for ticks
        "pgf.rcfonts": False,  # don't setup fonts from rc parameters
        "pgf.preamble": "\\usepackage{siunitx}",  # load additional packages
    }
)


# Create folder to store output
def get_out_folder(iteration: int | None, potential_type: PotentialType) -> Path:
    base = Path(f"{Path(__file__).parent}/out/{potential_type.to_string()}")
    if iteration is not None:
        out = base / f"{iteration:04}"
    else:
        for i in range(9999):
            out = base / f"{i:04}"
            if not out.exists():
                break
        else:
            raise ValueError("Every possible path already occupied")
    out.mkdir(parents=True, exist_ok=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fits a single time step to parameters"
    )
    parser.add_argument(
        "-i",
        "--iteration",
        type=int,
        default=None,
        help="Use existing output folder instead of creating new one",
    )
    parser.add_argument(
        "-p",
        "--potential_type",
        type=int,
        default=PotentialType.Mie,
        help="The interaction potential to use. Can be 0 for Morse or 1 for Mie.",
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
    potential_type = PotentialType(pyargs.potential_type)
    # potential_type: PotentialType = PotentialType.Mie

    out = get_out_folder(pyargs.iteration, potential_type)

    interval = time.time()
    mask0 = np.loadtxt(Path(__file__).parent / "image001032-markers.csv", delimiter=",")
    img0 = imread(str(Path(__file__).parent / "image001032.png"))
    mask1 = np.loadtxt(Path(__file__).parent / "image001042-markers.csv", delimiter=",")
    img1 = imread(str(Path(__file__).parent / "image001042.png"))
    mask2 = np.loadtxt(Path(__file__).parent / "image001052-markers.csv", delimiter=",")
    img2 = imread(str(Path(__file__).parent / "image001052.png"))
    n_vertices = 8

    # pos0 = np.array(crm.extract_positions(mask0, n_vertices))
    pos1, lengths1, radii1 = crm.extract_positions(mask1, n_vertices)
    pos2, lengths2, radii2 = crm.extract_positions(mask2, n_vertices)

    # Calculate Rod lengths which is later used to determine growth rates.
    rod_length_diffs = lengths2 - lengths1
    radii = (radii1 + radii2) / 2

    print(f"{time.time() - interval:10.4f}s Loaded data and calculated initial values")
    interval = time.time()

    # Fix some parameters
    domain_size = np.max(mask1.shape)
    cutoff = 20.0
    rigidity = 8.0

    # Fix some parameters for the simulation
    args_predict = (
        cutoff,
        rigidity,
        rod_length_diffs,
        domain_size,
        pos1,
        pos2,
        potential_type,
        out,
    )

    # Parameters
    damping = 1.5
    strength = 1.0

    # Bounds
    bounds = [
        [0.6, 3.0],  # Damping
        [0.5, 3.5],  # Strength
        *[[4.0, 12.0]] * len(radii),  # Radii
    ]

    if potential_type is PotentialType.Morse:
        # Parameter Values
        potential_stiffness = 0.4
        parameters = (*radii, damping, strength, potential_stiffness)

        # Constraints
        bounds.append([0.25, 0.55])  # Potential Stiffness
        A = np.zeros((len(bounds),) * 2)
        constraints = sp.optimize.LinearConstraint(A, lb=-np.inf, ub=np.inf)
    elif potential_type is PotentialType.Mie:
        # Parameter Values
        en = 6.0
        em = 5.5
        parameters = (*radii, damping, strength, en, em)

        # Constraints
        bounds.append([0.2, 25.0])  # en
        bounds.append([0.2, 25.0])  # em
        A = np.zeros((len(bounds),) * 2)
        A[0][len(bounds) - 2] = -1
        A[0][len(bounds) - 1] = 1
        lb = -np.inf
        ub = np.full(len(bounds), np.inf)
        ub[0] = -1
        constraints = sp.optimize.LinearConstraint(A, lb=lb, ub=ub)
    else:
        raise ValueError("potential type needs to be variant of PotentialType enum")

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
            x0=parameters,
            args=args_predict,
            workers=-1,
            updating="deferred",
            maxiter=20,
            # constraints=constraints,
            disp=True,
            tol=1e-4,
            recombination=0.3,
            popsize=50,
            polish=False,
            rng=0,
        )
        final_cost = res.fun
        final_params = res.x
        # Store information in file
        store_parameters(final_params, filename, out, final_cost)
        print(f"{time.time() - interval:10.4f}s Finished Parameter Optimization")

    interval = time.time()

    param_infos = [
        *[
            (f"Radius {i}", "\\SI{}{\\micro\\metre}", f"r_{{{i}}}")
            for i in range(len(radii))
        ],
        ("Damping", "\\SI{}{\\per\\min}", "\\lambda"),
        ("Strength", "\\SI{}{\\micro\\metre^2\\per\\min^2}", "C"),
    ]
    if potential_type is PotentialType.Morse:
        param_infos.append(
            ("Potential Stiffness", "\\SI{}{\\micro\\metre}", "\\lambda")
        )
    elif potential_type is PotentialType.Mie:
        param_infos.append(("Exponent n", "1", "n"))
        param_infos.append(("Exponent m", "1", "m"))

    # Plot Cost function against varying parameters
    if not pyargs.skip_profiles:
        pool = mp.Pool()
        for n, (p, bound) in enumerate(zip(final_params, bounds)):
            f_a = None
            f_a = plot_profile(
                n,
                bound,
                args_predict[:-1],
                param_infos[n],
                final_params,
                final_cost,
                out,
                pool,
                f_a,
            )
            fig, _ = f_a
            plt.close(fig)

        print(f"{time.time() - interval:10.4f} Plotted Profiles")
        interval = time.time()

    cell_container = predict(
        final_params,
        cutoff,
        rigidity,
        rod_length_diffs,
        pos1,
        domain_size,
        potential_type,
    )

    if cell_container is None:
        print("Best fit does not produce valid result.")
        exit()

    iterations = cell_container.get_all_iterations()
    agents_initial = cell_container.get_cells_at_iteration(iterations[0])
    agents_predicted = cell_container.get_cells_at_iteration(iterations[-1])

    if not pyargs.skip_masks:
        figs_axs = [plt.subplots() for _ in range(4)]
        figs_axs[0][1].imshow(img1)
        figs_axs[0][1].set_axis_off()
        figs_axs[1][1].imshow(img2)
        figs_axs[1][1].set_axis_off()
        figs_axs[2][1].imshow(mask1)
        figs_axs[2][1].set_axis_off()
        figs_axs[3][1].imshow(mask2)
        figs_axs[3][1].set_axis_off()

        for p in pos1:
            figs_axs[0][1].plot(p[:, 0], p[:, 1], color="white")
        for agent, _ in agents_predicted.values():
            p = agent.pos
            figs_axs[1][1].plot(p[:, 0], p[:, 1], color="white")

        for i, (fig, _) in enumerate(figs_axs):
            fig.tight_layout()
            fig.savefig(f"{out}/microscopic-images-{i}.png")

        mask_gen1 = crm.render_mask(
            agents_initial, cell_container.cell_to_color, domain_size
        )
        mask_gen2 = crm.render_mask(
            agents_predicted, cell_container.cell_to_color, domain_size
        )

        print(f"{time.time() - interval:10.4f} Rendered Masks")
        interval = time.time()

    if not pyargs.skip_param_space:
        visualize_param_space(out, param_infos, final_params, final_cost)
        print(f"{time.time() - interval:10.4f} Visualized parameter space")
        interval = time.time()

    if not pyargs.skip_distributions:
        plot_distributions(agents_predicted, out)
        print(f"{time.time() - interval:10.4f} Plotted Distributions")
