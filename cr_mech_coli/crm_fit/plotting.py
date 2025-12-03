import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path
from tqdm.contrib.concurrent import process_map
import cr_mech_coli as crm
from cr_mech_coli.cr_mech_coli import MorsePotentialF32
import scipy as sp

from cr_mech_coli.plotting import COLOR3, COLOR5

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
    all_params, bounds_lower, bounds_upper, n, param_single, args, pyargs = opt_args

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
            "maxiter": pyargs.profiles_maxiter,
            "maxfev": pyargs.profiles_maxiter,
        },
    )
    return res.fun


def plot_profile(
    n: int,
    args: tuple[np.ndarray, list[int], Settings],
    optimization_result: OptimizationResult,
    out: Path,
    n_workers,
    displacement_error: float,
    pyargs,
    fig_ax=None,
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

    (name, units, short) = param_info

    odir = out / "profiles"
    odir.mkdir(parents=True, exist_ok=True)

    x = np.linspace(bound_lower, bound_upper, pyargs.profiles_samples)
    savename = name.strip().lower().replace(" ", "-")
    try:
        y = np.loadtxt(odir / f"profile-{savename}")
    except:
        pool_args = [
            (
                optimization_result.params,
                infos.bounds_lower,
                infos.bounds_upper,
                n,
                p,
                args,
                pyargs,
            )
            for p in x
        ]

        y = process_map(
            optimize_around_single_param,
            pool_args,
            desc=f"Profile: {name}",
            max_workers=n_workers,
        )
        np.savetxt(odir / f"profile-{savename}", y)

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
        0,
        marker="x",
        color=COLOR5,
        alpha=0.7,
        s=12**2,
    )

    y = (y - final_cost) / displacement_error**2

    # Fill confidence levels
    thresh_prev = 0
    for i, q in enumerate([0.68, 0.90, 0.95]):
        thresh = sp.stats.chi2.ppf(q, 1)
        color = crm.plotting.COLOR3 if i % 2 == 0 else crm.plotting.COLOR5
        filt = y <= thresh
        lower = np.max(np.array([y, np.repeat(thresh_prev, len(y))]), axis=0)
        ax.fill_between(
            x,
            lower,
            np.repeat(thresh, len(lower)),
            where=filt,
            interpolate=True,
            color=color,
            alpha=0.3,
        )
        thresh_prev = thresh

    crm.plotting.configure_ax(ax)
    ax.plot(
        x,
        # (y - final_cost) / displacement_error**2,
        y,
        color=crm.plotting.COLOR3,
        linestyle="--",
    )

    upper = np.min([4 * thresh_prev, 1.05 * np.max([np.max(y), thresh_prev])])
    lower = -0.05 * upper
    ax.set_ylim(lower, upper)

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


def plot_distribution(n, name, values, out, infos):
    fig, ax = plt.subplots(figsize=(8, 8))
    crm.configure_ax(ax)
    ax.hist(values, color=COLOR3, edgecolor=COLOR3, alpha=0.6)

    bound_lower = infos.bounds_lower[n]
    bound_upper = infos.bounds_upper[n]
    param_info = infos.parameter_infos[n]
    (_, units, short) = param_info

    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_title(name)
    ax.set_xlabel(f"{short} [{units}]")
    ax.set_ylabel("Count")
    ax.set_xlim(bound_lower, bound_upper)

    savename = name.lower().replace(" ", "-")

    fig.savefig(out / f"{savename}.png")
    fig.savefig(out / f"{savename}.pdf")
    fig.clf()
