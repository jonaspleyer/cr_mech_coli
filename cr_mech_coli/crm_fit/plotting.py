import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm.contrib.concurrent import process_map
import cr_mech_coli as crm

from .crm_fit_rs import Settings, OptimizationResult, predict_calculate_cost


def pred_flatten_wrapper(args):
    parameters, iterations, positions_all, settings = args
    return predict_calculate_cost(parameters, positions_all, iterations, settings)


def plot_profile(
    n: int,
    args: tuple[list[int], np.ndarray, Settings],
    optimization_result: OptimizationResult,
    out: Path,
    n_workers,
    fig_ax=None,
    steps: int = 20,
):
    (_, positions_all, settings) = args
    infos = settings.generate_optimization_infos(positions_all.shape[1])
    bound_lower = infos.bounds_lower[n]
    bound_upper = infos.bounds_upper[n]
    param_info = infos.parameter_infos[n]

    if fig_ax is None:
        fig_ax = plt.subplots(figsize=(8, 8))
        fig, ax = fig_ax
    else:
        fig, ax = fig_ax
        fig.clf()

    x = np.linspace(bound_lower, bound_upper, steps)
    ps = [
        [pi if n != i else xi for i, pi in enumerate(optimization_result.params)]
        for xi in x
    ]

    (name, units, short) = param_info

    pool_args = [(p, *args) for p in ps]
    y = process_map(
        pred_flatten_wrapper, pool_args, desc=f"Profile: {name}", max_workers=n_workers
    )

    final_params = optimization_result.params
    final_cost = optimization_result.cost

    # Extend x and y by values from final_params and final cost
    x = np.append(x, final_params[n])
    y = np.append(y, final_cost)
    sorter = np.argsort(x)
    x = x[sorter]
    y = y[sorter]

    ax.set_title(name)
    ax.set_ylabel("Cost function L")
    ax.set_xlabel(f"{short} [{units}]")
    ax.scatter(
        final_params[n],
        final_cost,
        marker="o",
        edgecolor=crm.plotting.COLOR3,
        facecolor=crm.plotting.COLOR2,
    )
    crm.plotting.configure_ax(ax)
    ax.plot(x, y, color=crm.plotting.COLOR3, linestyle="--")
    fig.tight_layout()
    plt.savefig(f"{out}/profile-{name}.png".lower().replace(" ", "-"))
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


def plot_distributions(agents_predicted, out: Path):
    agents = [a[0] for a in agents_predicted.values()]
    growth_rates = np.array([a.growth_rate for a in agents])
    fig, ax = plt.subplots(figsize=(8, 8))
    ax2 = ax.twiny()
    ax.hist(
        growth_rates,
        edgecolor="k",
        linestyle="--",
        fill=None,
        label="Growth Rates",
        hatch=".",
    )
    ax.set_xlabel("Growth Rate [µm/min]")
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
    ax2.set_xlabel("Radius [µm]")
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
    fig.savefig(out / "growth_rates_lengths_distribution.png")
    fig.clf()
