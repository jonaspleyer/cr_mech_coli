import enum
import numpy as np
import cr_mech_coli as crm
from pathlib import Path


class PotentialType(enum.Enum):
    Morse = 0
    Mie = 1

    def to_string(self):
        if self is PotentialType.Morse:
            return "morse"
        elif self is PotentialType.Mie:
            return "mie"


def reconstruct_morse_potential(parameters, cutoff):
    (*radii, damping, strength, potential_stiffness) = parameters
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


def reconstruct_mie_potential(parameters, cutoff):
    (*radii, damping, strength, en, em) = parameters
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
    positions: np.ndarray,  # Shape (N, n_vertices, 3)
    domain_size: float,
    potential_type: PotentialType,
    out_path: Path | None = None,
):
    if potential_type is PotentialType.Morse:
        damping, interactions = reconstruct_morse_potential(parameters, cutoff)
    elif potential_type is PotentialType.Mie:
        damping, interactions = reconstruct_mie_potential(parameters, cutoff)

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
        if out_path is not None:
            with open(out_path / "logs.txt", "a+") as f:
                params_fmt = ",".join([f"{p}" for p in parameters])
                message = f"Error DURING SIMULATION\n{e}\nPARAMETERS:\n[{params_fmt}]\n"
                f.write(message)
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
        pos_initial,
        domain_size,
        potential_type,
        out_path,
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
