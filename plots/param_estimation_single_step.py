from cv2 import imread
import cr_mech_coli as crm
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import time
import enum


class PotentialType(enum.Enum):
    Morse = 0
    Mie = 1

    def to_string(self):
        if self is PotentialType.Morse:
            return "morse"
        elif self is PotentialType.Mie:
            return "mie"


def predict(
    # Parameters
    growth_rates: list[float],  # Shape (N)
    rigidity: float,
    interaction,
    # Constants
    positions: np.ndarray,  # Shape (N, n_vertices, 3)
    domain_size: float,
):
    config = crm.Configuration(
        domain_size=domain_size,
    )
    config.dt *= 0.25
    n_agents = positions.shape[0]
    n_vertices = positions.shape[1]

    def pos_to_spring_length(pos):
        res = np.sum(np.linalg.norm(pos[1:] - pos[:-1], axis=1)) / n_vertices
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
            interaction=interaction,
            growth_rate=growth_rates[i],
            spring_length=pos_to_spring_length(positions[i]),
            spring_length_threshold=1000,
            rigidity=rigidity,
        )
        for i in range(n_agents)
    ]
    return crm.run_simulation_with_agents(config, agents)


def reconstruct_morse_potential(parameters, cutoff):
    (*growth_rates, rigidity, radius, strength, potential_stiffness) = parameters
    interaction = crm.MorsePotentialF32(
        radius=radius,
        potential_stiffness=potential_stiffness,
        cutoff=cutoff,
        strength=strength,
    )
    return (growth_rates, rigidity, interaction)


def reconstruct_mie_potential(parameters, cutoff):
    (*growth_rates, rigidity, radius, strength, en, em) = parameters
    interaction = crm.MiePotentialF32(
        radius=radius,
        strength=strength,
        bound=4 * strength,
        cutoff=cutoff,
        en=en,
        em=em,
    )
    return (growth_rates, rigidity, interaction)


def predict_flatten(
    parameters: tuple,
    cutoff,
    domain_size,
    pos_initial,
    pos_final,
    potential_type: PotentialType = PotentialType.Morse,
    out_path: str | None = None,
    return_cells: bool = False,
):
    if potential_type is PotentialType.Morse:
        growth_rates, rigidity, interaction = reconstruct_morse_potential(
            parameters, cutoff
        )
    elif potential_type is PotentialType.Mie:
        growth_rates, rigidity, interaction = reconstruct_mie_potential(
            parameters, cutoff
        )
    cell_container = predict(
        growth_rates,
        rigidity,
        interaction,
        pos_initial,
        domain_size,
    )

    if return_cells:
        return cell_container

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
        out = ""
        for p in parameters:
            out += f"{p},"
        out += f"{cost}\n"
        with open(f"{out_path}/param-costs.csv", "a") as f:
            f.write(out)

    return cost


if __name__ == "__main__":
    import os

    interval = time.time()
    # markers = np.fromfile("./data/growth-2-marked/image001042-markers.tif").reshape(576, 768)
    # mask0 = np.loadtxt("data/growth-2-marked/image001032-markers.csv", delimiter=",")
    # img0 = imread("data/growth-2/image001032.png")
    mask1 = np.loadtxt("data/growth-2-marked/image001042-markers.csv", delimiter=",")
    img1 = imread("data/growth-2/image001042.png")
    mask2 = np.loadtxt("data/growth-2-marked/image001052-markers.csv", delimiter=",")
    img2 = imread("data/growth-2/image001052.png")
    n_vertices = 8
    # pos0 = np.array(crm.extract_positions(mask0, n_vertices))
    pos1 = np.array(crm.extract_positions(mask1, n_vertices))
    pos2 = np.array(crm.extract_positions(mask2, n_vertices))

    figs_axs = [plt.subplots() for _ in range(4)]
    figs_axs[0][1].imshow(img1)
    figs_axs[0][1].set_axis_off()
    figs_axs[1][1].imshow(img2)
    figs_axs[1][1].set_axis_off()
    figs_axs[2][1].imshow(mask1)
    figs_axs[2][1].set_axis_off()
    figs_axs[3][1].imshow(mask2)
    figs_axs[3][1].set_axis_off()

    print("{:8.3}s Generated initial plots".format(time.time() - interval))
    interval = time.time()

    domain_size = np.max(mask1.shape)
    cutoff = 30.0
    potential_type: PotentialType = PotentialType.Mie

    # Create folder to store output
    out = f"out/parameter-estimation/{potential_type.to_string()}"
    os.makedirs(out, exist_ok=True)

    args = (cutoff, domain_size, pos1, pos2, potential_type, out)

    growth_rates = [0.03] * pos1.shape[0]
    radius = 6.0
    strength = 0.2
    rigidity = 0.8
    if potential_type is PotentialType.Morse:
        potential_stiffness = 0.4
        parameters = (*growth_rates, rigidity, radius, strength, potential_stiffness)
    elif potential_type is PotentialType.Mie:
        en = 6.0
        em = 5.5
        parameters = (*growth_rates, rigidity, radius, strength, en, em)

    # Optimize values
    bounds = [
        *[[0.00, 0.1]] * pos1.shape[0],  # Growth Rates
        [0.4, 3.0],  # Rigidity
        [5.0, 7.0],  # Radius
        [0.1, 1.0],  # Strength
    ]
    if potential_type is PotentialType.Morse:
        bounds.append([0.25, 0.55])  # Potential Stiffness
        A = np.zeros((len(bounds),) * 2)
        constraints = sp.optimize.LinearConstraint(A, lb=-np.inf, ub=np.inf)
    elif potential_type is PotentialType.Mie:
        bounds.append([3.0, 30.0])  # en
        bounds.append([3.0, 30.0])  # em
        A = np.zeros((len(bounds),) * 2)
        A[0][len(bounds) - 2] = -1
        A[0][len(bounds) - 1] = 1
        lb = -np.inf
        ub = np.full(len(bounds), np.inf)
        ub[0] = -1
        constraints = sp.optimize.LinearConstraint(A, lb=lb, ub=ub)

    res = sp.optimize.differential_evolution(
        predict_flatten,
        bounds=bounds,
        x0=parameters,
        args=args,
        workers=-1,
        updating="deferred",
        maxiter=200,
        constraints=constraints,
        disp=True,
        tol=1e-4,
        recombination=0.3,
        popsize=200,
        polish=False,
    )
    print("{:8.4}s Finished Parameter Optimization".format(time.time() - interval))
    interval = time.time()

    param_infos = [
        *[(f"Growth Rate {i}", "\\mu m\\text{min}^{-1}") for i in range(pos1.shape[0])],
        ("Rigidity", "\\mu m\\text{min}^{-1}"),
        ("Radius", "\\mu m"),
        ("Strength", "\\mu m^2\text{min}^{-2}"),
    ]
    if potential_type is PotentialType.Morse:
        param_infos.append(("Potential Stiffness", "\\mu m"))
    elif potential_type is PotentialType.Mie:
        param_infos.append(("Exponent n", "1"))
        param_infos.append(("Exponent m", "1"))

    # Plot Cost function against varying parameters
    for n, (p, bound) in enumerate(zip(res.x, bounds)):
        fig2, ax2 = plt.subplots()

        x = np.linspace(bound[0], bound[1], 20)
        ps = [[pi if n != i else xi for i, pi in enumerate(res.x)] for xi in x]
        y = [predict_flatten(p, *args) for p in ps]

        (name, units) = param_infos[n]

        ax2.set_title(name)
        ax2.set_ylabel("Cost function $L$")
        ax2.set_xlabel("Parameter Value [${}$]".format(units))
        ax2.scatter(p, res.fun, marker="o", color="red")
        ax2.plot(x, y)
        fig2.tight_layout()
        plt.savefig(
            "docs/source/_static/fitting-methods/estimate-parameters/{}/{}.png".format(
                potential_type.to_string(), name
            )
        )
        plt.close(fig2)

    print("{:8.3} Plotted Profiles".format(time.time() - interval))
    interval = time.time()

    cell_container = predict_flatten(
        res.x,
        *args,
        return_cells=True,
    )

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
        fig.savefig(
            "docs/source/_static/fitting-methods/estimate-parameters/{}/microscopic-images-{}.png".format(
                potential_type.to_string(),
                i,
            )
        )

    print("{:8.3} Rendered Masks".format(time.time() - interval))
