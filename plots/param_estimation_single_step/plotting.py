import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
from pathlib import Path

from predict import predict_flatten


def plot_profile(
    n: int,
    bound,
    args: tuple,
    param_info: tuple,
    final_params,
    final_cost: float,
    out: Path,
    pool,
    fig_ax=None,
    steps: int = 20,
):
    if pool is None:
        pool = mp.Pool()
    if fig_ax is None:
        fig_ax = plt.subplots()
        fig, ax = fig_ax
    else:
        fig, ax = fig_ax
        fig.clf()

    x = np.linspace(bound[0], bound[1], steps)
    ps = [[pi if n != i else xi for i, pi in enumerate(final_params)] for xi in x]

    pool_args = [(p, *args) for p in ps]
    y = pool.starmap(predict_flatten, pool_args)
    # y = [predict_flatten(*pa) for pa in pool_args]

    (name, units, _) = param_info
    ax.set_title(name)
    ax.set_ylabel("Cost function $L$")
    ax.set_xlabel(f"Parameter Value [${units}$]")
    ax.scatter(
        final_params[n],
        final_cost,
        marker="o",
        edgecolor="k",
        facecolor=(0.3, 0.3, 0.3),
    )
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
    img = ax.imshow(np.abs(basis.T), cmap="Grays", vmin=0, vmax=1)
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

    radii = np.array([a.radius for a in agents])
    ax2.hist(
        radii,
        edgecolor="gray",
        linestyle="-",
        facecolor="gray",
        alpha=0.5,
        label="Radii",
    )
    ax2.set_xlabel("Interaction Thickness [$\\mu m$]")
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
    fig.savefig(out / "growth_rates_lengths_distribution.png")
    fig.clf()
