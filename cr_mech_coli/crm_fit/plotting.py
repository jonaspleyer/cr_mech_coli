import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm.contrib.concurrent import process_map
import cr_mech_coli as crm
from cr_mech_coli.cr_mech_coli import MorsePotentialF32
import scipy as sp
from tqdm import tqdm

from .crm_fit_rs import Settings, OptimizationResult, predict_calculate_cost


def pred_flatten_wrapper(args):
    parameters, iterations, positions_all, settings = args
    return predict_calculate_cost(parameters, positions_all, iterations, settings)


def prediction_optimize_helper(
    params_opt, param_single, n_param, positions_all, iterations, settings
):
    params_all = [0] * (len(params_opt) + 1)
    params_all[:n_param] = params_opt[:n_param]
    params_all[n_param] = param_single
    params_all[n_param + 1 :] = params_opt[n_param:]

    return predict_calculate_cost(params_all, positions_all, iterations, settings)


def optimize_around_single_param(opt_args):
    all_params, bounds_lower, bounds_upper, n, param_single, args = opt_args

    params_opt = list(all_params)
    b_low = list(bounds_lower)
    b_upp = list(bounds_upper)

    del params_opt[n]
    del b_low[n]
    del b_upp[n]

    bounds = [(b_low[i], b_upp[i]) for i in range(len(b_low))]

    res = sp.optimize.minimize(
        prediction_optimize_helper,
        x0=params_opt,
        args=(param_single, n, *args),
        bounds=bounds,
        method="Nelder-Mead",
        options={
            "disp": True,
            "maxiter": 10,
            "maxfev": 10,
        },
    )
    return res.fun


def plot_profile(
    n: int,
    args: tuple[np.ndarray, list[int], Settings],
    optimization_result: OptimizationResult,
    out: Path,
    n_workers,
    fig_ax=None,
    steps: int = 20,
):
    (positions_all, iterations, settings) = args
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

    (name, units, short) = param_info

    pool_args = [
        (optimization_result.params, infos.bounds_lower, infos.bounds_upper, n, p, args)
        for p in x
    ]

    y = process_map(
        optimize_around_single_param,
        pool_args,
        desc=f"Profile: {name}",
        max_workers=n_workers,
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
    odir = out / "profiles"
    odir.mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{odir}/{name}.png".lower().replace(" ", "-"))
    plt.savefig(f"{odir}/{name}.pdf".lower().replace(" ", "-"))
    return (fig, ax)


def plot_interaction_potential(
    settings: Settings,
    optimization_result: OptimizationResult,
    n_agents,
    out,
):
    if settings.parameters.potential_type == MorsePotentialF32:
        return None

    agent_index = 0
    expn = settings.get_param("Exponent n", optimization_result, n_agents, agent_index)
    expm = settings.get_param("Exponent m", optimization_result, n_agents, agent_index)
    radius = settings.get_param("Radius", optimization_result, n_agents, agent_index)
    strength = settings.get_param(
        "Strength", optimization_result, n_agents, agent_index
    )
    bound = settings.get_param("Bound", optimization_result, n_agents, agent_index)

    def mie_potential(x: np.ndarray):
        c = expn / (expn - expm) * (expn / expm) ** (expm / (expn - expm))
        sigma = radius * (expm / expn) ** (1 / (expn - expm))
        return np.minimum(
            strength * c * ((sigma / x) ** expn - (sigma / x) ** expm),
            np.array([bound] * len(x)),
        )

    x = np.linspace(0.05 * radius, settings.constants.cutoff, 200)
    y = mie_potential(x)

    fig, ax = plt.subplots(figsize=(8, 8))
    crm.plotting.configure_ax(ax)

    ax.plot(x / radius, y / strength, label="Mie Potential", color=crm.plotting.COLOR3)
    ax.set_xlabel("Distance [R]")
    ax.set_ylabel("Normalized Interaction Strength")

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=1,
        frameon=False,
    )

    fig.savefig(out / "potential-shape.png")
    fig.savefig(out / "potential-shape.pdf")


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
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=3, frameon=False)
    fig.savefig(out / "growth_rates_lengths_distribution.png")
    fig.savefig(out / "growth_rates_lengths_distribution.pdf")
    fig.clf()
