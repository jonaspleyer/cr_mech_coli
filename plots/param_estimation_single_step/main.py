from cv2 import imread
import cr_mech_coli as crm
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import time
import enum
from pathlib import Path
import multiprocessing as mp


class PotentialType(enum.Enum):
    Morse = 0
    Mie = 1

    def to_string(self):
        if self is PotentialType.Morse:
            return "morse"
        elif self is PotentialType.Mie:
            return "mie"


def reconstruct_morse_potential(parameters, radii, cutoff):
    (damping, strength, potential_stiffness) = parameters
    interactions = [
        crm.MorsePotentialF32(
            radius=r,
            potential_stiffness=potential_stiffness,
            cutoff=cutoff,
            strength=strength,
        )
        for r in radii
    ]
    return (damping, interactions)


def reconstruct_mie_potential(parameters, radii, cutoff):
    (damping, strength, en, em) = parameters
    interactions = [
        crm.MiePotentialF32(
            radius=r,
            strength=strength,
            bound=2 * strength,
            cutoff=cutoff,
            en=en,
            em=em,
        )
        for r in radii
    ]
    return (damping, interactions)


def predict(
    parameters,
    cutoff,
    rigidity,
    rod_length_diffs,
    radii,
    positions: np.ndarray,  # Shape (N, n_vertices, 3)
    domain_size: float,
    potential_type: PotentialType,
):
    if potential_type is PotentialType.Morse:
        damping, interactions = reconstruct_morse_potential(parameters, radii, cutoff)
    elif potential_type is PotentialType.Mie:
        damping, interactions = reconstruct_mie_potential(parameters, radii, cutoff)

    config = crm.Configuration(
        domain_size=domain_size,
        dt=0.02,
    )
    config.dt *= 0.25
    n_agents = positions.shape[0]
    n_vertices = positions.shape[1]

    def pos_to_spring_length(pos):
        res = np.sum(np.linalg.norm(pos[1:] - pos[:-1], axis=1)) / (n_vertices - 1)
        return res

    agents = [
        crm.RodAgent(
            pos=np.array(
                [*positions[i].T, [config.domain_height / 2] * positions.shape[1]],
                dtype=np.float32,
            ).T,
            vel=np.zeros(
                (positions.shape[1], positions.shape[2] + 1), dtype=np.float32
            ),
            interaction=interactions[i],
            growth_rate=rod_length_diffs[i] / config.t_max / (n_vertices - 1),
            spring_length=pos_to_spring_length(positions[i]),
            spring_length_threshold=1000,
            rigidity=rigidity,
            damping=damping,
        )
        for i in range(n_agents)
    ]
    try:
        return crm.run_simulation_with_agents(config, agents)
    except ValueError as e:
        print(f"{e}\nParameters used were:\n{parameters}")
        return None


def store_parameters(parameters, filename, out_path, cost=None):
    out_path.mkdir(parents=True, exist_ok=True)
    out = ""
    for p in parameters:
        out += f"{p},"
    if cost is not None:
        out += f"{cost}\n"
    with open(out_path / filename, "a+") as f:
        f.write(out)


def predict_flatten(
    parameters: tuple | list,
    cutoff,
    rigidity,
    rod_length_diffs,
    radii,
    domain_size,
    pos_initial,
    pos_final,
    potential_type: PotentialType,
    out_path: Path | None = None,
):
    cell_container = predict(
        parameters,
        cutoff,
        rigidity,
        rod_length_diffs,
        radii,
        pos_initial,
        domain_size,
        potential_type,
    )

    if cell_container is None:
        cost = np.inf
    else:
        final_iter = cell_container.get_all_iterations()[-1]
        final_cells = cell_container.get_cells_at_iteration(final_iter)
        final_cells = [(k, final_cells[k]) for k in final_cells]
        final_cells.sort(key=lambda x: x[0][1])
        pos_predicted = np.array([(kv[1][0]).pos for kv in final_cells])

        # TODO
        # This is currently very inefficient.
        # We could probably better match the
        # positions to each other
        cost = np.sum(
            [
                (pos_predicted[i][:, :2] - pos_final[i]) ** 2
                for i in range(len(pos_predicted))
            ]
        )

    if out_path is not None:
        store_parameters(parameters, "param-costs.csv", out_path, cost)

    return cost


def plot_profile(
    bound, args, param_info, final_params, final_cost, out, pool, fig_ax=None
):
    if pool is None:
        pool = mp.Pool()
    if fig_ax is None:
        fig_ax = plt.subplots()
        fig, ax = fig_ax
    else:
        fig, ax = fig_ax
        fig.clf()

    x = np.linspace(bound[0], bound[1], 20)
    ps = [[pi if n != i else xi for i, pi in enumerate(final_params)] for xi in x]

    pool_args = [(p, *args) for p in ps]
    y = pool.starmap(predict_flatten, pool_args)
    # y = [predict_flatten(*pa) for pa in pool_args]

    (name, units, _) = param_info
    ax.set_title(name)
    ax.set_ylabel("Cost function $L$")
    ax.set_xlabel(f"Parameter Value [${units}$]")
    ax.scatter(p, final_cost, marker="o", edgecolor="k", facecolor=(0.3, 0.3, 0.3))
    ax.plot(x, y, color="k", linestyle="--")
    fig.tight_layout()
    plt.savefig(f"{out}/{name}.png")
    return (fig, ax)


def _get_orthogonal_basis_by_cost(parameters, p0, costs, c0):
    ps = parameters - p0
    # Calculate geometric mean of differences
    # dps = np.abs(ps).prod(axis=1) ** (1.0 / ps.shape[1])
    dps = np.linalg.norm(ps, axis=1)
    dcs = costs - c0

    # Filter any values with smaller costs
    filt = (dcs >= 0) * (dps > 0)
    ps = ps[filt]
    dps = dps[filt]
    dcs = dcs[filt]

    # Calculate gradient of biggest cost
    dcs_dps = dcs / dps
    ind = np.argmax(dcs_dps)
    basis = [ps[ind] / np.linalg.norm(ps[ind])]
    contribs = [dcs_dps[ind]]

    for _ in range(len(p0) - 1):
        # Calculate orthogonal projection along every already obtained basis vector
        ortho = ps
        for b in basis:
            ortho = ortho - np.outer(np.sum(ortho * b, axis=1) / np.sum(b**2), b)
        factors = np.linalg.norm(ortho, axis=1) / np.linalg.norm(ps, axis=1)
        dcs *= factors
        dcs_dps = dcs / dps
        ind = np.argmax(dcs_dps)
        basis.append(ortho[ind] / np.linalg.norm(ortho[ind]))
        contribs.append(dcs_dps[ind])
    return np.array(basis), np.array(contribs) / np.sum(contribs)


def visualize_param_space(out: Path, param_infos, params=None):
    if params is None:
        params = np.genfromtxt(out / "final_params.csv", delimiter=",")
    params = np.array(params)
    param_costs = np.genfromtxt(out / "param-costs.csv", delimiter=",")

    basis, contribs = _get_orthogonal_basis_by_cost(
        param_costs[:, :-1], params[:-1], param_costs[:, -1], params[-1]
    )

    # Plot matrix
    fig, ax = plt.subplots()
    names = [f"${p[2]}$" for p in param_infos]
    img = ax.imshow(np.abs(basis.T), cmap="Grays")
    plt.colorbar(img, ax=ax)
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(
        [
            f"$\\vec{{v}}_{{{i}}}~{contribs[i] * 100:5.1f}\\%$"
            for i in range(len(param_infos))
        ]
    )
    ax.set_yticklabels(names)
    ax.set_ylabel("Parameters")
    ax.set_xlabel("Basis Vectors")

    print(basis.shape)
    try:
        print("Rank:", np.linalg.matrix_rank(basis))
    except:
        print("Calculation of rank failed.")
    fig.savefig(out / "parameter_space_matrix.png")


def plot_distributions(agents_predicted, out: Path):
    agents = [a[0] for a in agents_predicted.values()]
    growth_rates = np.array([a.growth_rate for a in agents])
    fig, ax = plt.subplots()
    ax2 = ax.twiny()
    ax.hist(
        growth_rates,
        edgecolor="k",
        linestyle="--",
        fill=None,
        label="Growth Rates",
        hatch=".",
    )
    ax.set_xlabel("Growth Rate [$\\mu\\text{min}^{-1}$]")
    ax.set_ylabel("Count")

    lengths = np.array([np.linalg.norm(a.pos[1:] - a.pos[:-1]) for a in agents])
    ax2.hist(
        lengths,
        edgecolor="gray",
        linestyle="-",
        facecolor="gray",
        alpha=0.5,
        label="Rod Lengths",
    )
    ax2.set_xlabel("Rod Length [$\\mu m$]")
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
    fig.savefig(out / "growth_rates_lengths_distribution.png")
    fig.clf()


if __name__ == "__main__":
    interval = time.time()
    mask0 = np.loadtxt(Path(__file__).parent / "image001032-markers.csv", delimiter=",")
    img0 = imread(Path(__file__).parent/"image001032.png")
    mask1 = np.loadtxt(Path(__file__).parent / "image001042-markers.csv", delimiter=",")
    img1 = imread(Path(__file__).parent / "image001042.png")
    mask2 = np.loadtxt(Path(__file__).parent / "image001052-markers.csv", delimiter=",")
    img2 = imread(Path(__file__).parent / "image001052.png")
    n_vertices = 8
    # pos0 = np.array(crm.extract_positions(mask0, n_vertices))
    pos1, lengths1, radii1 = crm.extract_positions(mask1, n_vertices)
    pos2, lengths2, radii2 = crm.extract_positions(mask2, n_vertices)

    # Claculate Rod lengths which is later used to determine growth rates.
    rod_length_diffs = lengths2 - lengths1
    radii = (radii1 + radii2) / 2

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
    potential_type: PotentialType = PotentialType.Mie

    # Create folder to store output
    out = Path(f"out/parameter-estimation/{potential_type.to_string()}")
    out.mkdir(parents=True, exist_ok=True)

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
        param_infos.append(("Potential Stiffness", "\\mu m"))
    elif potential_type is PotentialType.Mie:
        param_infos.append(("Exponent n", "1", "n"))
        param_infos.append(("Exponent m", "1", "m"))

    # Plot Cost function against varying parameters
    pool = mp.Pool()
    for n, (p, bound) in enumerate(zip(final_params, bounds)):
        f_a = None
        f_a = plot_profile(
            bound, args, param_infos[n], final_params, final_cost, out, pool, f_a
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

    visualize_param_space(Path("out/parameter-estimation/mie"), param_infos)
    plot_distributions(agents_predicted, out)

    print(f"{time.time() - interval:10.4f} Visualized parameter space")
