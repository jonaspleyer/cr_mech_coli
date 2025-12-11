import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import scipy as sp

import cr_mech_coli as crm
from cr_mech_coli import crm_fit


def plot_potential_single(n, name, short, units, path, ax, color=crm.plotting.COLOR3):
    savename = name.strip().lower().replace(" ", "-")
    y = np.loadtxt(path / f"profiles/profile-{savename}")

    settings = crm_fit.Settings.from_toml(path / "settings.toml")
    optimization_result = crm_fit.OptimizationResult.load_from_file(
        path / "final_params.toml"
    )

    settings = crm_fit.Settings.from_toml(path / "settings.toml")
    optim_res = crm_fit.OptimizationResult.load_from_file(path / "final_params.toml")
    p_fixed = optim_res.params[n]
    final_cost = optim_res.cost
    displacement_error = settings.constants.displacement_error

    infos = settings.generate_optimization_infos(6)
    bound_lower = infos.bounds_lower[n]
    bound_upper = infos.bounds_upper[n]

    try:
        x = np.load(path / f"profiles/profile-{savename}-params")
    except:
        x = np.linspace(bound_lower, bound_upper, len(y))

    crm_fit.plot_profile_from_data(
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
    )

    return ax, x, (y - final_cost) / displacement_error**2


def plot_single_profile_combination(name, short, units, n1_p1, n2_p2, odir):
    crm.plotting.set_mpl_rc_params()
    fig, ax = plt.subplots(figsize=(8, 8))
    crm.plotting.configure_ax(ax)

    x1 = []
    y1 = []
    if type(n1_p1) is tuple:
        n1, p1 = n1_p1
        res1 = plot_potential_single(
            n1,
            name,
            short,
            units,
            p1,
            ax,
            color=crm.plotting.COLOR3,
        )
        x1 = res1[1]
        y1 = res1[2]

    x2 = []
    y2 = []
    if type(n2_p2) is tuple:
        n2, p2 = n2_p2
        res2 = plot_potential_single(
            n2,
            name,
            short,
            units,
            p2,
            ax,
            color=crm.plotting.COLOR5,
        )
        x2 = res2[1]
        y2 = res2[2]

    x_unique = np.sort(np.unique([*x1, *x2]))
    x = []
    y = []
    for xi in x_unique:
        n1 = np.where(xi >= x1)[0][-1]
        n2 = np.where(xi >= x2)[0][-1]

        yi1 = y1[n1]
        yi2 = y2[n2]

        if yi1 < yi2 and xi in x1:
            x.append(xi)
            y.append(yi1)
        if yi2 < yi1 and xi in x2:
            x.append(xi)
            y.append(yi2)

    x = np.array(x)
    y = np.array(y)

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

    odir.mkdir(parents=True, exist_ok=True)

    savename = name.strip().lower().replace(" ", "-")
    fig.savefig(odir / f"profile-{savename}.pdf")
    fig.savefig(odir / f"profile-{savename}.png")
    plt.close(fig)


def plot_all_profiles_combined(p1, p2, odir):
    settings1 = crm_fit.Settings.from_toml(p1 / "settings.toml")
    settings2 = crm_fit.Settings.from_toml(p2 / "settings.toml")

    infos1 = settings1.generate_optimization_infos(6)
    infos2 = settings2.generate_optimization_infos(6)

    infos_combined = []
    for n1, (name, short, units) in enumerate(infos1.parameter_infos):
        infos_combined.append(((n1, p1), None, name, short, units))
    for n2, (name, short, units) in enumerate(infos2.parameter_infos):
        found = False
        for n, info in enumerate(infos_combined):
            if name == info[2]:
                infos_combined[n] = (info[0], (n2, p2), info[2], info[3], info[4])
                found = True
        if found is False:
            pass
            # infos_combined.append(((None, (n2, p2), (name, short, units))))

    for n1_p1, n2_p2, name, short, units in infos_combined:
        plot_single_profile_combination(
            name,
            short,
            units,
            n1_p1,
            n2_p2,
            odir,
        )


if __name__ == "__main__":
    path_mie_all = Path("figures/crm_fit/mie_all")
    path_mie_partial = Path("figures/crm_fit/mie_partial")
    path_morse_all = Path("figures/crm_fit/morse_all")
    path_morse_partial = Path("figures/crm_fit/morse_partial")

    plot_all_profiles_combined(
        path_mie_all,
        path_mie_partial,
        Path("figures/crm_fit/mie_profiles"),
    )

    plot_all_profiles_combined(
        path_morse_all,
        path_morse_partial,
        Path("figures/crm_fit/morse_profiles"),
    )
