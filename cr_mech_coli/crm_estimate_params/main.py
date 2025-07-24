from itertools import repeat
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cr_mech_coli as crm
from cr_mech_coli.plotting import COLOR3, COLOR5
import scipy as sp
import multiprocessing as mp
from pathlib import Path
from glob import glob


def delayed_growth(t, x0, growth_rate, t0):
    x = np.array(t < t0)
    return x * x0 + ~x * x0 * np.exp((t - t0) * growth_rate)


def confidence_region(popt, pcov, ax, n_std=1.0, **kwargs):
    pearson = pcov[0, 1] / np.sqrt(pcov[0, 0] * pcov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    radius_x = np.sqrt(1 + pearson)
    radius_y = np.sqrt(1 - pearson)
    ellipse = mpl.patches.Ellipse(
        (0, 0),
        width=radius_x * 2,
        height=radius_y * 2,
        **kwargs,
    )

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(pcov[0, 0]) * n_std

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(pcov[1, 1]) * n_std

    transf = (
        mpl.transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(popt[0], popt[1])
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def extract_pos(args):
    i, filename, n_vertices = args
    m = np.loadtxt(filename, delimiter=",").T
    try:
        return (i, crm.extract_positions(m, n_vertices)[0])
    except:
        print(f"Could not extract position at iteration {i:06}")
        return None


def estimate_growth_curves_individual(filenames, out_path, delay=None):
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    n_vertices = 12

    pool = mp.Pool()

    args = list(zip(range(len(filenames)), filenames, repeat(n_vertices)))

    results = pool.map(extract_pos, args)
    results = list(filter(lambda x: x is not None, results))
    iterations = np.array([r[0] for r in results])
    positions = np.array([r[1] for r in results])

    rod_lengths = np.sum(
        np.linalg.norm(positions[:, :, 1:] - positions[:, :, :-1], axis=3), axis=2
    )

    y = np.mean(rod_lengths, axis=1)
    yerr = np.std(rod_lengths, axis=1)

    # Prepare Figure
    crm.plotting.set_mpl_rc_params()
    fig, ax = plt.subplots(figsize=(8, 8))
    crm.plotting.configure_ax(ax)

    # Set Labels
    ax.set_xlabel("Time [frames]")
    ax.set_ylabel("Rod Length [pix]")

    # Plot Data
    ax.plot(y, color=COLOR3, label="Data")
    ax.fill_between(iterations, y - yerr, y + yerr, color=COLOR3, alpha=0.3)

    if delay is None:
        growth_curve = delayed_growth
        p0 = (y[0], np.log(y[-1] / y[0]), len(iterations) / 2)
    else:

        def special_delayed_growth(t, x0, growth_rate):
            return delayed_growth(t, x0, growth_rate, delay)

        growth_curve = special_delayed_growth
        p0 = (y[0], np.log(y[-1] / y[0]))

    # Plot Exponential Fit
    popt, pcov = sp.optimize.curve_fit(
        growth_curve,
        iterations,
        y,
        p0=p0,
        sigma=yerr,
        absolute_sigma=True,
    )
    ax.plot(
        iterations,
        growth_curve(iterations, *popt),
        color=COLOR5,
        linestyle="--",
        label="Fit",
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=2,
        frameon=False,
    )
    popt_mean = popt
    pcov_mean = pcov

    fig.tight_layout()
    fig.savefig(out_path / "rod-lengths-average.png")
    fig.savefig(out_path / "rod-lengths-average.pdf")

    ax.cla()
    parameters = []
    covariances = []
    crm.plotting.configure_ax(ax)
    for i in range(rod_lengths.shape[1]):
        yi = rod_lengths[:, i]
        if delay is None:
            p0 = (yi[0], np.log(yi[-1] / yi[0]), len(iterations) / 2)
        else:
            p0 = (yi[0], np.log(yi[-1] / yi[0]))

        popt, pcov = sp.optimize.curve_fit(
            growth_curve,
            iterations,
            yi,
            p0=p0,
        )
        parameters.append(popt)
        covariances.append(pcov)
        ax.plot(iterations, yi, color=COLOR3, label="Data")
        ax.plot(
            iterations,
            growth_curve(iterations, *popt),
            label="Fit",
            color=COLOR5,
            linestyle="--",
        )

    ax.set_xlabel("Time [frames]")
    ax.set_ylabel("Rod Length [pix]")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[:2],
        labels[:2],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=2,
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(out_path / "rod-lengths-individual.png")
    fig.savefig(out_path / "rod-lengths-individual.pdf")

    ax.cla()
    crm.plotting.configure_ax(ax)

    parameters = np.array(parameters)
    x = parameters[:, 0]
    if delay is None:
        y = parameters[:, 2]
    else:
        y = parameters[:, 1]
    ax.scatter(x, y, color=COLOR3)

    for popt, pcov in zip(parameters, covariances):
        if delay is None:
            pm = popt[[0, 2]]
            pc = pcov[0:3:2, 0:3:2]
        else:
            pm = popt[:2]
            pc = pcov[:2, :2]

        confidence_region(pm, pc, ax, color=COLOR3, alpha=0.3, label="Individual")

    if delay is None:
        pm = popt_mean[[0, 2]]
        pc = pcov_mean[0:3:2, 0:3:2]
    else:
        pm = popt_mean[:2]
        pc = pcov_mean[:2, :2]
    ax.scatter([pm[0]], [pm[1]], color=COLOR5)

    confidence_region(pm, pc, ax, color=COLOR5, alpha=0.3, label="Mean")

    if delay is None:
        ax.set_xlabel("Delay [frame]")
        ax.set_ylabel("Growth Rate [1/frame]")
    else:
        ax.set_xlabel("Starting Length [pix]")
        ax.set_ylabel("Growth Rate [1/frame]")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        [handles[0], handles[-1]],
        [labels[0], labels[-1]],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=2,
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(out_path / "parameter-distribution.png")
    fig.savefig(out_path / "parameter-distribution.pdf")


def crm_estimate_params_main():
    filenames = list(sorted(glob("data/crm_fit/0004/masks/*.csv")))
    estimate_growth_curves_individual(
        filenames, "out/crm_estimate_params/IWF-Goettingen/"
    )

    filenames = [
        f"data/raw/2007-youtube/markers/{i:06}-markers.csv" for i in range(20, 27)
    ]
    estimate_growth_curves_individual(
        filenames,
        "out/crm_estimate_params/2007-youtube/",
        delay=0,
    )
