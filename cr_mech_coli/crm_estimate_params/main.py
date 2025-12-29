from itertools import repeat
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cr_mech_coli as crm
from cr_mech_coli.plotting import COLOR2, COLOR3, COLOR4, COLOR5
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
    m = np.loadtxt(filename, delimiter=",", converters=float).T
    try:
        return (i, crm.extract_positions(m, n_vertices))
    except:
        print(f"Could not extract position at iteration {i:06}")
        return None


def estimate_growth_curves_individual(
    filenames,
    out_path,
    delay=None,
    pixel_per_micron=None,
    minutes_per_frame=None,
    use_positions=True,
):
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    n_vertices = 12

    if use_positions:
        masks = [np.loadtxt(f, delimiter=",", converters=float) for f in filenames]
        results = [crm.extract_positions(m, n_vertices) for m in masks]
        inds = [r[3] for r in results]
        rod_lengths = np.array([r[1][np.argsort(i)] for r, i in zip(results, inds)])
    else:
        masks = [
            np.loadtxt(f, delimiter=",", converters=float).astype(float)
            for f in filenames
        ]
        counts = [np.unique_counts(m) for m in masks]
        rod_lengths = []
        for c in counts:
            ind = np.argsort(c.values)
            filt = c.values[ind] != 0
            rod_lengths.append(c.counts[ind][filt])
        rod_lengths = np.array(rod_lengths)

    t = [int(f.split("/")[-1].split(".csv")[0].split("-")[0]) for f in filenames]
    t = np.array(t, dtype=float) - np.min(t).astype(float)
    y = np.mean(rod_lengths, axis=1)
    yerr = np.std(rod_lengths, axis=1)

    if pixel_per_micron is not None:
        y /= pixel_per_micron
        yerr /= pixel_per_micron
    if minutes_per_frame is not None:
        t *= minutes_per_frame

    # Prepare Figure
    crm.plotting.set_mpl_rc_params()
    fig, ax = plt.subplots(figsize=(8, 8))
    crm.plotting.configure_ax(ax)

    # Set Labels
    ax.set_xlabel("Time [frames]")
    if use_positions and pixel_per_micron is not None:
        ax.set_ylabel("Rod Length [µm]")
    elif not use_positions:
        ax.set_ylabel("Pixels per Rod [counts]")
    else:
        ax.set_ylabel("Rod Length [pix]")

    # Plot Data
    ax.plot(t, y, color=COLOR3, label="Data")
    ax.fill_between(t, y - yerr, y + yerr, color=COLOR3, alpha=0.3)

    if delay is None:
        growth_curve = delayed_growth
        p0 = (y[0], np.log(y[-1] / y[0]), np.max(t) / 2)
    else:

        def special_delayed_growth(t, x0, growth_rate):
            return delayed_growth(t, x0, growth_rate, delay)

        growth_curve = special_delayed_growth
        p0 = (y[0], np.log(y[-1] / y[0]))

    # Plot Exponential Fit
    popt, pcov = sp.optimize.curve_fit(
        growth_curve,
        t,
        y,
        p0=p0,
        sigma=yerr,
        absolute_sigma=True,
    )
    ax.plot(
        t,
        growth_curve(t, *popt),
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
        if pixel_per_micron is not None:
            yi /= pixel_per_micron
        if delay is None:
            p0 = (yi[0], np.log(yi[-1] / yi[0]), np.max(t) / 2)
        else:
            p0 = (yi[0], np.log(yi[-1] / yi[0]))

        popt, pcov = sp.optimize.curve_fit(
            growth_curve,
            t,
            yi,
            p0=p0,
        )
        parameters.append(popt)
        covariances.append(pcov)
        ax.plot(t, yi, color=COLOR3, label="Data")
        ax.plot(
            t,
            growth_curve(t, *popt),
            label="Fit",
            color=COLOR5,
            linestyle="--",
        )

    if minutes_per_frame is not None:
        ax.set_xlabel("Time [min]")
    else:
        ax.set_xlabel("Time [frames]")
    if pixel_per_micron is not None:
        ax.set_ylabel("Rod Length [µm]")
    elif not use_positions:
        ax.set_ylabel("Pixels per Rod [counts]")
    else:
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

    growth_rates = [p[1] for p in parameters]
    growth_rates_uncert = [p[1][1] ** 0.5 for p in covariances]

    xmin = np.min(np.array(growth_rates) - 2 * np.array(growth_rates_uncert))
    xmax = np.max(np.array(growth_rates) + 2 * np.array(growth_rates_uncert))

    x = np.linspace(xmin, xmax, 200)
    dx = (xmax - xmin) / len(x)

    yfin = np.zeros(x.shape)
    for n in range(len(growth_rates)):
        mu = growth_rates[n]
        sigma = growth_rates_uncert[n]
        y = (
            1
            / np.sqrt(2 * np.pi * sigma**2)
            * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
        )
        yfin += y
        ax.plot(x, y, color=COLOR2, alpha=0.6, linestyle="--", label="Individual Fits")
    yfin /= len(growth_rates)

    # Plot curve fillings for 20% intervals
    z = np.array([np.sum(yfin[:i]) for i in range(len(yfin))]) * dx

    ind0 = 0
    for n, q in enumerate(np.arange(0, 1.2, 0.2)):
        ind1 = np.argmin(q >= z)
        if ind1 == 0:
            ind1 = len(z)
        ind2 = min(ind1 + 1, len(z))
        ax.fill_between(
            x[ind0:ind2],
            np.zeros(ind2 - ind0),
            yfin[ind0:ind2],
            color=COLOR2 if n % 2 == 0 else COLOR4,
            alpha=0.5,
        )
        ind0 = ind1

    i0 = np.argmin(0.5 >= z)
    i1 = i0 + 1
    q = z[i0] - 0.5
    xmean = q * x[i0] + (1 - q) * x[i1]
    ymean = q * yfin[i0] + (1 - q) * yfin[i1]

    ax.plot(x, yfin, color=COLOR3, label="Sum")
    ax.vlines(xmean, 0, ymean, color="red")

    if minutes_per_frame is not None:
        ax.set_xlabel("Growth Rate [1/frame]")
    else:
        ax.set_xlabel("Growth Rate [1/min]")
    ax.set_ylabel("Probability")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        [handles[0], handles[-1]],
        [labels[0], labels[-1]],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=2,
        frameon=False,
    )

    fig.savefig(out_path / "growth-rates-with-uncert.png")
    fig.savefig(out_path / "growth-rates-with-uncert.pdf")
    ax.cla()

    crm.plotting.configure_ax(ax, minor=False)

    ax.hist(growth_rates, facecolor=COLOR3, edgecolor=COLOR2)
    xmin = np.min(growth_rates)
    xmax = np.max(growth_rates)
    middle = (xmin + xmax) / 2
    dx = xmax - xmin
    xticks = [xmin + 0.05 * dx, middle, xmax - 0.05 * dx]
    ax.set_xticks(xticks, labels=[f"{x:f}" for x in xticks])
    ax.set_xlabel("Growth Rate [1/min]")
    ax.set_ylabel("Count")

    np.savetxt(out_path / "growth_rates.csv", growth_rates, delimiter=",")

    yticks = ax.get_yticks()
    yticks = list(filter(lambda x: int(x) == x, yticks))
    ax.set_yticks(yticks, minor=False)

    fig.savefig(out_path / "growth-rates-bar-plot.png")
    fig.savefig(out_path / "growth-rates-bar-plot.pdf")
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
        if pixel_per_micron is not None:
            ax.set_xlabel("Starting Length [pix]")
        else:
            ax.set_xlabel("Starting Length [µm]")
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
    filenames = list(sorted(glob("data/crm_fit/mie_all/masks/*.csv")))
    estimate_growth_curves_individual(
        filenames,
        "out/crm_estimate_params/IWF-Goettingen/positions/",
        delay=0,
        pixel_per_micron=15.0,
        minutes_per_frame=20 / 8,
    )

    estimate_growth_curves_individual(
        filenames,
        "out/crm_estimate_params/IWF-Goettingen/pixel-count/",
        delay=0,
        minutes_per_frame=20 / 8,
        use_positions=False,
    )

    filenames = [
        f"data/raw/2007-youtube/markers/{i:06}-markers.csv" for i in range(20, 27)
    ]
    estimate_growth_curves_individual(
        filenames,
        "out/crm_estimate_params/2007-youtube/",
        delay=0,
    )
