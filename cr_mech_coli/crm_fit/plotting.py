import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
from pathlib import Path
from tqdm.contrib.concurrent import process_map

from .predict import predict_flatten


def pred_flatten_wrapper(args):
    return predict_flatten(*args)


def plot_profile(
    n: int,
    bound,
    args: tuple,
    param_info: tuple,
    final_params,
    final_cost: float,
    out: Path,
    n_workers,
    fig_ax=None,
    steps: int = 20,
):
    if fig_ax is None:
        fig_ax = plt.subplots()
        fig, ax = fig_ax
    else:
        fig, ax = fig_ax
        fig.clf()

    x = np.linspace(bound[0], bound[1], steps)
    ps = [[pi if n != i else xi for i, pi in enumerate(final_params)] for xi in x]

    (name, units, short) = param_info

    pool_args = [(p, *args) for p in ps]
    y = process_map(
        pred_flatten_wrapper, pool_args, desc=f"Profile: {name}", max_workers=n_workers
    )

    # Extend x and y by values from final_params and final cost
    x = np.append(x, final_params[n])
    y = np.append(y, final_cost)
    sorter = np.argsort(x)
    x = x[sorter]
    y = y[sorter]

    ax.set_title(name)
    ax.set_ylabel("Cost function $L$")
    ax.set_xlabel(f"${short}$ $[{units}]$")
    ax.scatter(
        final_params[n],
        final_cost,
        marker="o",
        edgecolor="k",
        facecolor=(0.3, 0.3, 0.3),
    )
    ax.plot(x, y, color="k", linestyle="--")
    fig.tight_layout()
    plt.savefig(f"{out}/{name}.png".lower().replace(" ", "-"))
    return (fig, ax)


def _get_orthogonal_basis_by_cost(parameters, p0, costs, c0):
    ps = parameters / p0 - 1
    # Calculate geometric mean of differences
    # dps = np.abs(ps).prod(axis=1) ** (1.0 / ps.shape[1])
    dps = np.linalg.norm(ps, axis=1)
    dcs = costs - c0
    ps_norms = np.linalg.norm(ps, axis=1)

    # Filter any values with smaller costs
    filt = (dcs >= 0) * (dps > 0) * np.isfinite(dps) * np.isfinite(dcs)
    ps = ps[filt]
    dps = dps[filt]
    dcs = dcs[filt]
    ps_norms = ps_norms[filt]

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
        factors = np.linalg.norm(ortho, axis=1) / ps_norms
        dcs *= factors
        dcs_dps = dcs / dps
        ind = np.argmax(dcs_dps)
        basis.append(ortho[ind] / np.linalg.norm(ortho[ind]))
        contribs.append(dcs_dps[ind])
    return np.array(basis), np.array(contribs) / np.sum(contribs)


def visualize_param_space(out: Path, param_infos, final_params, final_cost):
    final_params = np.array(final_params)
    param_costs = np.genfromtxt(out / "param-costs.csv", delimiter=",")

    basis, contribs = _get_orthogonal_basis_by_cost(
        param_costs[:, :-1],
        final_params,
        param_costs[:, -1],
        final_cost,
    )

    # Plot matrix
    fig, ax = plt.subplots()
    names = [f"${p[2]}$" for p in param_infos]
    img = ax.imshow(np.abs(basis), cmap="Grays", vmin=0, vmax=1)
    plt.colorbar(img, ax=ax)
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticklabels(
        [
            f"$\\vec{{v}}_{{{i}}}~{contribs[i] * 100:5.1f}\\%$"
            for i in range(len(param_infos))
        ]
    )
    ax.set_xticklabels(names)
    ax.set_xlabel("Parameters $\\vec{p}$")
    ax.set_ylabel("Basis Vectors")

    try:
        rank = f"\nwith rank {np.linalg.matrix_rank(basis)}/{np.min(basis.shape)}"
    except:
        print("Calculation of rank failed.")
        rank = ""
    ax.set_title(
        "Gradient of Cost $\\vec{\\nabla}_{\\vec{v}_i} L(\\vec{p}_\\text{opt})$" + rank
    )
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
    ax.set_xlabel("Growth Rate [$\\SI{}{\\micro\\metre\\per\\min}$]")
    ax.set_ylabel("Count")

    radii = np.array([a.radius for a in agents])
    ax2.hist(
        radii,
        edgecolor="gray",
        linestyle="-",
        facecolor="gray",
        alpha=0.5,
        label="Radii",
    )
    ax2.set_xlabel("Radius [$\\SI{}{\\micro\\metre}$]")
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
    fig.savefig(out / "growth_rates_lengths_distribution.png")
    fig.clf()
