import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import scipy as sp

import cr_mech_coli as crm
from cr_mech_coli import crm_fit

from cr_mech_coli import COLOR2, COLOR3, COLOR5, COLOR6


def plot_potential_single(
    n,
    name,
    short,
    units,
    path,
    ax,
    label,
    color=crm.plotting.COLOR3,
    linestyle="--",
    try_use_filter=True,
):
    savename = name.strip().lower().replace(" ", "-")
    y = np.loadtxt(path / f"profiles/profile-{savename}")

    settings = crm_fit.Settings.from_toml(path / "settings.toml")
    optim_res = crm_fit.OptimizationResult.load_from_file(path / "final_params.toml")
    p_fixed = optim_res.params[n]
    final_cost = optim_res.cost
    displacement_error = settings.constants.displacement_error

    infos = settings.generate_optimization_infos(6)
    bound_lower = infos.bounds_lower[n]
    bound_upper = infos.bounds_upper[n]

    param_path = path / f"profiles/profile-{savename}-params"
    try:
        x = np.load(param_path)
    except:
        x = np.linspace(bound_lower, bound_upper, len(y))
        np.save(param_path, x)

    if try_use_filter:
        try:
            filt = np.load(path / f"profiles/profile-{savename}-filter.npy")
        except:
            filt = np.array([True] * len(x))
        x = x[filt]
        y = y[filt]

    x, y = crm_fit.plot_profile_from_data(
        x,
        y,
        p_fixed,
        final_cost,
        ax,
        name,
        short,
        units,
        displacement_error,
        ls_color=color,
        fill=False,
        label=label,
        linestyle=linestyle,
        filter_thresh=16 if name == "Strength" else 16,
    )

    return ax, x, y  # , (y - final_cost) / displacement_error**2


def combine_values(xs, ys) -> tuple[list, list]:
    xall = []
    for xi in xs:
        xall.extend(xi)

    xcombined = np.sort(np.unique(xall))

    # def interpolate(x1, x2, y1, y2, xmiddle):
    #     return (xmiddle - x1) / (x2 - x1) * (y2 - y1) + y1

    def interpolate(x, y, xnew):
        # calculate derivative at point xnew
        if xnew < x[0] or xnew > x[-1]:
            return np.inf
        else:
            i = np.sum(x <= xnew) - 1
            dt = np.clip(np.abs(xnew - x[i]) / (x[i + 1] - x[i]), 0, 1)
            return (1 - dt) * y[i] + dt * y[i + 1]

    # Create new y values
    yall = []
    for x, y in zip(xs, ys):
        ynew = []
        for xi in xcombined:
            (i_x,) = np.where(xi == x)
            if len(i_x) == 0:
                yi = interpolate(x, y, xi)
                ynew.append(yi)
            else:
                ynew.append(y[i_x[0]])
        yall.append(ynew)

    yall = np.min(yall, axis=0)

    return xcombined, yall


def plot_all_profiles_combined(*args, odir, bounds={}):
    settings = []
    infos = []
    for p, label, kwargs in args:
        s = crm_fit.Settings.from_toml(p / "settings.toml")
        settings.append(s)
        infos.append((s.generate_optimization_infos(6), kwargs))

    infos_combined = {}
    for (p, label, _), (info, kwargs) in zip(args, infos):
        for n, (name, short, units) in enumerate(info.parameter_infos):
            if name in infos_combined:
                infos_combined[name].append((n, p, label, name, short, units, kwargs))
            else:
                infos_combined[name] = [(n, p, label, name, short, units, kwargs)]

    for inf in infos_combined.values():
        crm.plotting.set_mpl_rc_params()
        fig, ax = plt.subplots(figsize=(8, 8))
        crm.plotting.configure_ax(ax)

        xall = []
        yall = []

        name = ""
        for n, p, label, name, short, units, kwargs in inf:
            ax, xi, yi = plot_potential_single(
                n,
                name,
                short,
                units,
                p,
                ax,
                label,
                **kwargs,
            )
            xall.append(xi)
            yall.append(yi)

        ncol = max(2, round(len(inf) / 2))
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15) if len(inf) >= 3 else (0.5, 1.125),
            ncol=ncol,
            frameon=False,
        )

        if name in bounds:
            xmin, xmax = bounds[name]
            dx = xmax - xmin
        else:
            xmin = np.min([np.min(xi) for xi in xall])
            xmax = np.max([np.max(xi) for xi in xall])
            dx = xmax - xmin
        ax.set_xlim(xmin - 0.05 * dx, xmax + 0.05 * dx)

        x, y = combine_values(xall, yall)
        crm_fit.fill_confidence_levels(x, y, ax)

        ax.set_title(name, pad=55.0)
        odir.mkdir(parents=True, exist_ok=True)
        savename = name.strip().replace(" ", "-").lower()
        fig.savefig(odir / f"profile-{savename}.png")
        fig.savefig(odir / f"profile-{savename}.pdf")


def plot_optimization_progressions_combined(*args, ylim=(3.5, 6)):
    crm.plotting.set_mpl_rc_params()
    fig, ax = plt.subplots(figsize=(8, 8))
    crm.plotting.configure_ax(ax)

    for p, label, linestyle, color in args:
        # Load results
        result = crm_fit.OptimizationResult.load_from_file(p / "final_params.toml")
        y = result.evals
        x = np.arange(1, len(y) + 1)
        ax.plot(x, y, label=label, color=color, linestyle=linestyle)

    ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.set_ylim(*ylim)

    ax.set_xlabel("Iterations")
    ax.set_ylabel("Cost Function")

    ncol = max(2, round(len(args) / 2))
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15) if len(args) >= 3 else (0.5, 1.125),
        ncol=ncol,
        frameon=False,
    )
    fig.savefig("figures/crm_fit/optimization-progression.png")
    fig.savefig("figures/crm_fit/optimization-progression.pdf")


if __name__ == "__main__":
    path_mie_all = Path("figures/crm_fit/mie_all")
    path_mie_partial = Path("figures/crm_fit/mie_partial")
    path_morse_all = Path("figures/crm_fit/morse_all")
    path_morse_partial = Path("figures/crm_fit/morse_partial")

    bounds = {
        "Strength": (0, 0.12),
        "Damping": (0, 25.0),
        "Radius": (0.1, 0.75),
    }

    plot_all_profiles_combined(
        (path_morse_all, "Morse", {"linestyle": "--", "color": COLOR2}),
        (
            path_morse_partial,
            "Morse λ=1min$^{-1}$",
            {"linestyle": "-.", "color": COLOR3},
        ),
        (path_mie_all, "Mie", {"linestyle": "--", "color": COLOR5}),
        (path_mie_partial, "Mie n=1", {"linestyle": "-.", "color": COLOR6}),
        odir=Path("figures/fit/profiles"),
        bounds=bounds,
    )

    plot_optimization_progressions_combined(
        (path_morse_all, "Morse", "--", COLOR2),
        (path_morse_partial, "Morse λ=1min$^{-1}$", "-.", COLOR3),
        (path_mie_all, "Mie", "--", COLOR5),
        (path_mie_partial, "Mie n=1", "-.", COLOR6),
    )
