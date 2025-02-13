from cv2 import imread
import cr_mech_coli as crm
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import time
from pathlib import Path
import multiprocessing as mp
import argparse

from plotting import plot_profile, plot_distributions, visualize_param_space
from predict import predict_flatten, predict, store_parameters, PotentialType


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
    args = parser.parse_args()
    potential_type = PotentialType(args.potential_type)
    # potential_type: PotentialType = PotentialType.Mie

    out = get_out_folder(args.iteration, potential_type)

    interval = time.time()
    mask0 = np.loadtxt(Path(__file__).parent / "image001032-markers.csv", delimiter=",")
    img0 = imread(Path(__file__).parent / "image001032.png")
    mask1 = np.loadtxt(Path(__file__).parent / "image001042-markers.csv", delimiter=",")
    img1 = imread(Path(__file__).parent / "image001042.png")
    mask2 = np.loadtxt(Path(__file__).parent / "image001052-markers.csv", delimiter=",")
    img2 = imread(Path(__file__).parent / "image001052.png")
    n_vertices = 8

    # pos0 = np.array(crm.extract_positions(mask0, n_vertices))
    pos1, lengths1, radii1 = crm.extract_positions(mask1, n_vertices)
    pos2, lengths2, radii2 = crm.extract_positions(mask2, n_vertices)

    # Calculate Rod lengths which is later used to determine growth rates.
    rod_length_diffs = lengths2 - lengths1
    radii = (radii1 + radii2) / 2

    print(f"{time.time() - interval:10.4f}s Generated initial plots")
    interval = time.time()

    figs_axs = [plt.subplots() for _ in range(4)]
    figs_axs[0][1].imshow(img1)
    figs_axs[0][1].set_axis_off()
    figs_axs[1][1].imshow(img2)
    figs_axs[1][1].set_axis_off()
    figs_axs[2][1].imshow(mask1)
    figs_axs[2][1].set_axis_off()
    figs_axs[3][1].imshow(mask2)
    figs_axs[3][1].set_axis_off()

    print(f"{time.time() - interval:10.4f}s Generated initial plots")
    interval = time.time()

    domain_size = np.max(mask1.shape)
    cutoff = 20.0

    # Fix some parameters for the simulation
    rigidity = 8.0
    args = (
        cutoff,
        rigidity,
        rod_length_diffs,
        radii,
        domain_size,
        pos1,
        pos2,
        potential_type,
        out,
    )

    # growth_rates = [0.03] * pos1.shape[0]
    damping = 1.5
    # radius = 6.0
    strength = 1.0
    if potential_type is PotentialType.Morse:
        potential_stiffness = 0.4
        parameters = (damping, strength, potential_stiffness)
    elif potential_type is PotentialType.Mie:
        en = 6.0
        em = 5.5
        parameters = (damping, strength, en, em)

    # Optimize values
    bounds = [
        # *[[0.01, 0.05]] * pos1.shape[0],  # Growth Rates
        [0.6, 3.0],  # Damping
        # [0.01, 1.5],  # Rigidity
        # [4, 10.0],  # Radius
        [0.5, 3.5],  # Strength
    ]
    if potential_type is PotentialType.Morse:
        bounds.append([0.25, 0.55])  # Potential Stiffness
        A = np.zeros((len(bounds),) * 2)
        constraints = sp.optimize.LinearConstraint(A, lb=-np.inf, ub=np.inf)
    elif potential_type is PotentialType.Mie:
        bounds.append([0.2, 25.0])  # en
        bounds.append([0.2, 25.0])  # em
        A = np.zeros((len(bounds),) * 2)
        A[0][len(bounds) - 2] = -1
        A[0][len(bounds) - 1] = 1
        lb = -np.inf
        ub = np.full(len(bounds), np.inf)
        ub[0] = -1
        constraints = sp.optimize.LinearConstraint(A, lb=lb, ub=ub)

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
            args=args,
            workers=-1,
            updating="deferred",
            maxiter=20,
            constraints=constraints,
            disp=True,
            tol=1e-4,
            recombination=0.3,
            popsize=50,
            polish=False,
        )
        final_cost = res.fun
        final_params = res.x
        print(f"{time.time() - interval:10.4f}s Finished Parameter Optimization")
        # Store information in file
        store_parameters(final_params, filename, out, final_cost)

    interval = time.time()

    param_infos = [
        # *[
        #     (f"Growth Rate {i}", "\\mu m\\text{min}^{-1}", f"\\mu_{{{i}}}")
        #     for i in range(pos1.shape[0])
        # ],
        # ("Rigidity", "\\mu m\\text{min}^{-1}"),
        ("Damping", "\\text{min}^{-1}", "\\lambda"),
        # ("Radius", "\\mu m", "r"),
        ("Strength", "\\mu m^2\\text{min}^{-2}", "C"),
    ]
    if potential_type is PotentialType.Morse:
        param_infos.append(("Potential Stiffness", "\\mu m", "\\lambda"))
    elif potential_type is PotentialType.Mie:
        param_infos.append(("Exponent n", "1", "n"))
        param_infos.append(("Exponent m", "1", "m"))

    # Plot Cost function against varying parameters
    pool = mp.Pool()
    for n, (p, bound) in enumerate(zip(final_params, bounds)):
        f_a = None
        f_a = plot_profile(
            n, bound, args, param_infos[n], final_params, final_cost, out, pool, f_a
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
        radii,
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

    mask_gen1 = crm.render_mask(
        agents_initial, cell_container.cell_to_color, domain_size
    )
    mask_gen2 = crm.render_mask(
        agents_predicted, cell_container.cell_to_color, domain_size
    )

    for p in pos1:
        figs_axs[0][1].plot(p[:, 0], p[:, 1], color="white")
    for agent, _ in agents_predicted.values():
        p = agent.pos
        figs_axs[1][1].plot(p[:, 0], p[:, 1], color="white")

    for i, (fig, _) in enumerate(figs_axs):
        fig.tight_layout()
        fig.savefig(f"{out}/microscopic-images-{i}.png")

    print(f"{time.time() - interval:10.4f} Rendered Masks")
    interval = time.time()

    visualize_param_space(out, param_infos)
    plot_distributions(agents_predicted, out)

    print(f"{time.time() - interval:10.4f} Visualized parameter space")
