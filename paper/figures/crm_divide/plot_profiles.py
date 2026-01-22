import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path
from tqdm import tqdm

import cr_mech_coli as crm


def plot_profile_single(
    samples: np.ndarray[tuple[int], np.dtype[np.float64]],
    costs: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    p_fixed: float,
    final_cost: tuple[float, float, float],
    name: str,
    units: str,
    xlow: float | None = None,
    xhigh: float | None = None,
):
    x = samples
    y1 = costs[:, 0]
    y2 = costs[:, 1]
    y3 = costs[:, 2]
    filt = costs[:, 0] != np.nan
    filt2 = costs[:, 0] <= 50_000
    filt *= filt2

    if xlow is not None:
        filt3 = x >= xlow
        filt *= filt3
    if xhigh is not None:
        filt4 = x <= xhigh
        filt *= filt4

    # Cost with overlap
    cwo = y2[filt]
    # Cost without overlap
    cwoup = y2[filt] - y3[filt]
    # Cost with overlap and parent penalty = 1
    cwopp1 = y1[filt] + y3[filt]

    x = np.array(x)[filt]

    # Add entry for final cost
    # Sort entries by value of the parameter

    cwo_fin = final_cost[1]
    cwoup_fin = final_cost[1] - final_cost[2]
    cwopp1_fin = final_cost[0] + final_cost[2]

    cwo = np.array([*cwo, cwo_fin])
    cwoup = np.array([*cwoup, cwoup_fin])
    cwopp1 = np.array([*cwopp1, cwopp1_fin])

    # Correctly sort new values
    x = np.array([*x, p_fixed])
    sorter = np.argsort(x)
    x = x[sorter]

    cwo = cwo[sorter]
    cwoup = cwoup[sorter]
    cwopp1 = cwopp1[sorter]

    fig, ax = plt.subplots(figsize=(8, 8))
    crm.plotting.configure_ax(ax)

    ax.plot(x, cwo, c=crm.plotting.COLOR3, label="Metric")
    ax.plot(x, cwoup, c=crm.plotting.COLOR3, linestyle="--", label="Overlaps p$_o$=0")
    ax.plot(x, cwopp1, c=crm.plotting.COLOR3, linestyle=":", label="Parents p$_p$=1")

    handles, labels = ax.get_legend_handles_labels()
    empty_handle = mpl.lines.Line2D([], [], alpha=0)
    ax.legend(
        labels=[labels[0], "", *labels[-2:]],
        handles=[handles[0], empty_handle, *handles[-2:]],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=2,
        frameon=False,
    )

    ax.scatter([p_fixed], [cwo_fin], c=crm.plotting.COLOR5, marker="x")
    ax.scatter([p_fixed], [cwoup_fin], c=crm.plotting.COLOR5, marker="x")
    ax.scatter([p_fixed], [cwopp1_fin], c=crm.plotting.COLOR5, marker="x")

    ax.set_xlabel(f"{name} [{units}]")
    ax.set_ylabel("Cost L(θ)")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 1))

    opath1 = Path("figures/crm_divide/profiles-pretty/pdf/")
    opath2 = Path("figures/crm_divide/profiles-pretty/png/")
    opath1.mkdir(exist_ok=True, parents=True)
    opath2.mkdir(exist_ok=True, parents=True)
    savename = name.replace(" ", "-").lower()
    fig.savefig(opath1 / f"profile-{savename}.pdf")
    fig.savefig(opath2 / f"profile-{savename}.png")
    plt.close(fig)


if __name__ == "__main__":
    samples = np.load("figures/crm_divide/profile-samples.npy")
    costs = np.load("figures/crm_divide/profile-costs.npy")

    assert costs.shape[:2] == samples.shape

    n_params = np.arange(samples.shape[1])
    param_infos = [
        ("Radius", "µm"),
        ("Strength", "µm^2/min^2"),
        ("Potential Stiffness", "µm"),
        ("Growth Rate 0", "1/min"),
        ("Growth Rate 1", "1/min"),
        ("Growth Rate 2", "1/min"),
        ("Growth Rate 3", "1/min"),
        ("Growth Rate 4", "1/min"),
        ("Growth Rate 5", "1/min"),
        ("Division Length 0", "µm"),
        ("Division Length 1", "µm"),
        ("Division Length 2", "µm"),
        ("Division Length 3", "µm"),
        ("Growth Rate 0-0", "1/min", 0.0, 0.06),
        ("Growth Rate 0-1", "1/min", 0.0, 0.06),
        ("Growth Rate 1-0", "1/min", 0.0, 0.06),
        ("Growth Rate 1-1", "1/min", 0.0, 0.06),
        ("Growth Rate 2-0", "1/min", 0.0, 0.06),
        ("Growth Rate 2-1", "1/min", 0.0, 0.06),
        ("Growth Rate 3-0", "1/min", 0.0, 0.06),
        ("Growth Rate 3-1", "1/min", 0.0, 0.06),
    ]

    # settings = crm_fit.Settings.from_toml("figures/crm_divide/settings.toml")
    result = np.loadtxt("figures/crm_divide/optimize_result.csv")
    evals = np.loadtxt("figures/crm_divide/optimization_evals.csv")
    final_costs = np.load("figures/crm_divide/optimization-final-costs.npy")

    crm.plotting.set_mpl_rc_params()

    for n, pinfo in tqdm(enumerate(param_infos), total=len(param_infos)):
        p_fixed = result[n]
        plot_profile_single(samples[:, n], costs[:, n], p_fixed, final_costs, *pinfo)
