import numpy as np
import cr_mech_coli as crm
from pathlib import Path


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
    positions: np.ndarray,  # Shape (N, n_vertices, 3)
    settings,
    out_path: Path | None = None,
) -> crm.CellContainer | None:
    try:
        return settings.predict(parameters, positions)
    except ValueError as e:
        if out_path is not None:
            with open(out_path / "logs.txt", "a+") as f:
                params_fmt = ",".join([f"{p}" for p in parameters])
                message = f"Error DURING SIMULATION\n{e}\nPARAMETERS:\n[{params_fmt}]\n"
                f.write(message)
        return None


def store_parameters(parameters, filename, out_path, cost=None):
    # TODO we should probably replace this by some method which is more efficient computationally
    # Reference to global buffer which is being emptied iteratively.
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
    pos_initial,
    pos_final,
    settings,
    out_path: Path | None = None,
):
    cell_container = predict(
        parameters,
        pos_initial,
        settings,
        out_path,
    )

    if cell_container is None:
        cost = np.inf
    else:
        final_iter = cell_container.get_all_iterations()[-1]
        final_cells = cell_container.get_cells_at_iteration(final_iter)
        final_cells = [(k, final_cells[k]) for k in final_cells]
        final_cells.sort(key=lambda x: x[0])
        pos_predicted = np.array(
            [(kv[1][0]).pos for kv in final_cells], dtype=np.float32
        )

        cost = np.sum(
            [
                (pos_predicted[i][:, :2] - pos_final[i]) ** 2
                for i in range(len(pos_predicted))
            ]
        )

    if out_path is not None:
        store_parameters(parameters, "param-costs.csv", out_path, cost)

    return cost
