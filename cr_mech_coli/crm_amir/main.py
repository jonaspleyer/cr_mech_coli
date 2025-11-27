from cr_mech_coli import crm_amir
import cr_mech_coli as crm
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy as sp
import skimage as sk
from tqdm import tqdm
from pathlib import Path
import multiprocessing as mp
import argparse
import itertools

from cr_mech_coli.plotting import COLOR2, COLOR3, COLOR5

GREEN_COLOR = np.array([95, 231, 76])

ERROR_COST = 1e6
PIXELS_PER_MICRON = 102 / 10


def calculate_angle(p: np.ndarray, parameters: crm_amir.Parameters) -> float:
    intersection = np.array([parameters.block_size, parameters.domain_size / 2.0])
    endpoint = p[-1] if p[-1, 0] >= p[0, 0] else p[0]
    if endpoint[0] < parameters.block_size:
        return np.nan
    l1 = np.linalg.norm(endpoint - intersection)
    segments = np.linalg.norm(p[1:] - p[:-1], axis=1)
    l2 = np.sum(segments) - parameters.block_size
    angle = np.acos(l1 / np.clip(l2, 0, np.inf))
    return angle


def generate_parameters(**kwargs) -> crm_amir.Parameters:
    parameters = crm_amir.Parameters()
    parameters.block_size = 25.0
    parameters.dt = 0.005
    parameters.t_max = 200
    parameters.domain_size = 400
    n_vertices = 20
    parameters.n_vertices = n_vertices
    parameters.growth_rate = 0.03 * 7 / (n_vertices - 1)
    parameters.rod_rigidity = 20.0 * n_vertices / 20
    parameters.save_interval = 20
    parameters.damping = 0.02
    parameters.spring_tension = 10.0
    parameters.drag_force = 0.001

    for k, v in kwargs:
        parameters.__setattr__(k, v)
    return parameters


def plot_angles_and_endpoints():
    parameters = generate_parameters()

    endpoints = []
    y_collection = []
    rod_rigidities = np.linspace(0.3, 30, 20, endpoint=True)
    for rod_rigidity in rod_rigidities:
        parameters.rod_rigidity = rod_rigidity
        agents = crm_amir.run_sim(parameters)
        t = np.array([a[0] for a in agents]) * parameters.dt

        angles = [
            calculate_angle(a[1].agent.pos[:, [0, 2]], parameters) for a in agents
        ]
        y_collection.append(np.column_stack([t, angles]))

        endpoints.append(np.array([a.agent.pos[-1, [0, 2]] for _, a in agents]))

    cmap = crm.plotting.cmap

    # Create line collection
    line_collection = mpl.collections.LineCollection(
        y_collection, array=rod_rigidities, cmap=cmap
    )
    y_collection = np.array(y_collection)

    # Prepare Figure
    fig, [ax1, ax2] = plt.subplots(ncols=2, figsize=(16, 8))

    # Define x and y limits
    y = y_collection[:, :, 1::2]
    t = y_collection[:, :, ::2][~np.isnan(y)]
    ax2.set_xlim(float(np.min(t)), float(np.max(t)))
    ylow = float(np.nanmin(y))
    yhigh = float(np.nanmax(y))
    ax2.set_ylim(ylow - 0.05 * (yhigh - ylow), yhigh + 0.05 * (yhigh - ylow))

    # Add curves
    ax2.add_collection(line_collection)

    ax2.set_ylabel("Angle [radian]")
    ax2.set_xlabel("Time [min]")
    yticks = [0, np.pi / 8, np.pi / 4, 3 * np.pi / 8, np.pi / 2]
    yticklabels = ["0", "π/8", "π/4", "3π/8", "π/2"]
    ax2.set_yticks(yticks, yticklabels)

    # 2nd Plot: Endpoints
    endpoints = (
        np.array([parameters.domain_size, 0])
        + np.array([-1, 1]) * np.array(endpoints)[:, :, ::-1]
    )
    line_collection = mpl.collections.LineCollection(
        endpoints,
        array=rod_rigidities,
        cmap=cmap,
    )
    ax1.add_collection(line_collection)

    # Define x and y limits
    ymax = np.max(endpoints[:, :, 1::2])
    xmax = np.max(np.abs(endpoints[:, :, ::2] - parameters.domain_size / 2.0))
    dmax = max(xmax, ymax)
    ax1.set_ylim(0, 1.2 * dmax)
    ax1.set_xlim(
        parameters.domain_size / 2 - 0.6 * dmax,
        parameters.domain_size / 2 + 0.6 * dmax,
    )
    ax1.fill_between(
        [0, parameters.domain_size],
        [0, 0],
        [parameters.block_size] * 2,
        color="k",
        alpha=0.3,
    )
    ax1.set_xlabel("[µm]")
    ax1.set_ylabel("[µm]")

    # Apply settings to axis
    crm.plotting.configure_ax(ax2)
    crm.plotting.configure_ax(ax1)

    # Save Figure
    # fig.tight_layout()
    fig.colorbar(line_collection, label="Rod Rigidity", ax=ax2)
    fig.savefig("out/crm_amir/angles-endpoints.pdf")
    fig.savefig("out/crm_amir/angles-endpoints.png")


def extract_mask(iteration, img, n_vertices: int, output_dir):
    img2 = np.copy(img)
    filt1 = img2[:, :, 1] <= 150
    img2[filt1] = [0, 0, 0]
    filt2 = np.all(img2 >= np.array([180, 180, 180]), axis=2)
    img2[filt2] = [0, 0, 0]

    cutoff = int(img2.shape[1] / 3)
    filt3 = np.linalg.norm(img2 - GREEN_COLOR, axis=2) >= 100
    filt3[:, :cutoff] = True
    img2[filt3] = [0, 0, 0]

    img3 = np.copy(img2)
    img3[filt3 == 0] = GREEN_COLOR.astype(np.uint8)

    img_filt = sk.segmentation.expand_labels(img3, distance=20)

    img3 = np.repeat(np.all(img_filt != [0, 0, 0], axis=2), 3).reshape(
        img_filt.shape
    ).astype(int) * GREEN_COLOR.astype(int)
    img4 = np.copy(img3).astype(np.uint8)

    try:
        pos = crm.extract_positions(img4, n_vertices)[0][0]
        p = pos[:, ::-1].reshape((-1, 1, 2))
        img4 = cv.polylines(
            np.copy(img),
            [p.astype(int)],
            isClosed=False,
            color=(10, 10, 230),
            thickness=2,
        )
        ret = pos
    except ValueError as e:
        print(e)
        ret = None

    odir = output_dir / "progressions"
    odir.mkdir(parents=True, exist_ok=True)
    cv.imwrite(odir / f"step1-{iteration:06}.png", img)
    cv.imwrite(odir / f"step2-{iteration:06}.png", img2)
    cv.imwrite(odir / f"step3-{iteration:06}.png", img3)
    cv.imwrite(odir / f"step4-{iteration:06}.png", img4)

    return ret


def calculate_x_shift(p, block_size):
    y = p[-1, 0] - p[:, 0]
    x = block_size <= y
    ind = np.argmin(x)
    s = (y[ind] - block_size) / (y[ind + 1] - y[ind])
    x_pos = (s * p[ind] + (1 - s) * p[ind + 1])[1]
    return x_pos


def objective_function(
    params,
    set_params: dict,
    positions_data,
    iterations_data,
    x0_bounds,
    return_all=False,
    print_output=False,
):
    parameters, t_relax = create_default_parameters(positions_data, iterations_data)

    for k, v in set_params.items():
        parameters.__setattr__(k, v)

    # Variable Parameters
    for name, value in zip(x0_bounds.keys(), params):
        parameters.__setattr__(name, value)

    try:
        rods = crm_amir.run_sim_with_relaxation(parameters, t_relax)
    except ValueError:
        if print_output:
            print(f"ERROR Returning {ERROR_COST}")
        return ERROR_COST

    # Get initial and final position of rod
    p_rods = np.array(
        [
            rods[0][1].agent.pos[:, np.array([0, 2])],
            rods[1][1].agent.pos[:, np.array([0, 2])],
            rods[2][1].agent.pos[:, np.array([0, 2])],
        ]
    )

    p_rods[:, :, 1] = parameters.domain_size - p_rods[:, :, 1]

    # Shift such that start points align
    positions_data = (
        np.array([parameters.domain_size, 0]) - np.array([1, -1]) * positions_data
    )
    positions_data = positions_data[:, ::-1]
    for i in range(positions_data.shape[0]):
        positions_data[i, 0, 0] -= positions_data[i, 0, 0]
    x_shift_positions0 = calculate_x_shift(positions_data[0], parameters.block_size)
    x_shift_p0 = calculate_x_shift(p_rods[0], parameters.block_size)
    x_shift_diff = x_shift_positions0 - x_shift_p0
    positions_data[:, :, 1] -= x_shift_diff

    if return_all:
        return p_rods, positions_data, parameters

    diff = p_rods[1:] - positions_data[1:]
    cost = np.sum(diff**2)

    if print_output:
        print(f"f(x)={cost:>7.4f}", end=" ")
        for name, p in zip(x0_bounds.keys(), params):
            print(f"{name}={p:.4f}", end=" ")
        print()

    return cost


def plot_results(
    popt,
    positions_data: np.ndarray,
    iterations_data,
    x0_bounds: dict,
    set_params,
    output_dir,
):
    p_rods, positions_data, parameters = objective_function(
        popt, set_params, positions_data, iterations_data, x0_bounds, return_all=True
    )
    p0 = p_rods[0]
    p1 = p_rods[1]

    #
    fig, ax = plt.subplots(figsize=(8, 8))
    diffs = np.linalg.norm(positions_data[1:] - p_rods[1:], axis=(1, 2))
    labels = ["Bending", "Relaxation"]
    b = ax.bar(labels, diffs, color=crm.plotting.COLOR3)
    ax.bar_label(
        b,
        [f"{100 * p:.2f}%%" for p in diffs / np.sum(diffs)],
        label_type="edge",
        color=crm.plotting.COLOR5,
        weight="bold",
    )
    ax.set_ylim(0, np.max(diffs) * 1.1)
    ax.set_title("Cost Contributions")
    ax.set_ylabel("Cost")
    fig.savefig(output_dir / "cost-contributions.png")
    fig.savefig(output_dir / "cost-contributions.pdf")
    plt.close(fig)

    # Plot Comparison of fit with positional data
    fig, ax = plt.subplots(figsize=(8, 8))
    crm.configure_ax(ax, minor=False)
    ax.plot(p0[:, 1], p0[:, 0], color=crm.plotting.COLOR2, linestyle=":")
    ax.plot(p1[:, 1], p1[:, 0], color=crm.plotting.COLOR3, linestyle=":")
    ax.plot(
        positions_data[0, :, 1],
        positions_data[0, :, 0],
        color=crm.plotting.COLOR2,
        linestyle="--",
        alpha=0.5,
    )
    x_shift = calculate_x_shift(positions_data[0], parameters.block_size)
    ax.scatter(x_shift, parameters.block_size, marker="x", color=crm.plotting.COLOR4)
    ax.plot(
        positions_data[1, :, 1],
        positions_data[1, :, 0],
        color=crm.plotting.COLOR3,
        linestyle="--",
        alpha=0.5,
    )

    # Define limits for plot
    dx = parameters.domain_size / 3
    ax.set_xlim(dx, parameters.domain_size - dx)
    ax.set_ylim(0.7 * parameters.block_size, parameters.domain_size - dx)
    ax.fill_between(
        [0, parameters.domain_size],
        [parameters.block_size] * 2,
        color="gray",
        alpha=0.4,
    )
    ax.set_xlabel("[µm]")
    ax.set_ylabel("[µm]")
    ax.set_title("Fit Comparison")
    fig.savefig(output_dir / "fit-comparison.png")
    fig.savefig(output_dir / "fit-comparison.pdf")
    plt.close(fig)

    t_relax = iterations_data[2] - iterations_data[1]
    agents = [x[1].agent for x in crm_amir.run_sim_with_relaxation(parameters, t_relax)]
    for a in agents:
        a.radius = np.float32(parameters.domain_size / 50)
    render_snapshots(agents, parameters, output_dir)


def calculate_profile_point(
    n: int,
    pnew: float,
    popt,
    positions_data: np.ndarray,
    iterations_data,
    x0_bounds: dict,
    set_params,
    pyargs,
):
    x0_bounds_new = {k: v for i, (k, v) in enumerate(x0_bounds.items()) if i != n}
    x0 = np.array([p for i, p in enumerate(popt) if i != n])
    bounds = np.array([(x[0], x[2]) for x in x0_bounds_new.values()])

    assert len(x0_bounds_new) + 1 == len(x0_bounds)
    assert len(x0) == len(x0_bounds_new)
    assert len(x0) == len(bounds)

    init = np.repeat(x0[np.newaxis, :], pyargs.popsize_profiles, axis=0)

    sampler = sp.stats.qmc.LatinHypercube(len(x0))
    sample = sampler.random(pyargs.popsize_profiles - 1)
    sample = bounds[:, 0] + sample * (bounds[:, 1] - bounds[:, 0])
    init = np.vstack([*sample, x0])

    res = sp.optimize.differential_evolution(
        objective_function,
        args=(
            set_params | {list(x0_bounds.keys())[n]: pnew},
            positions_data,
            iterations_data,
            x0_bounds_new,
        ),
        bounds=bounds,
        maxiter=pyargs.maxiter_profiles,
        popsize=pyargs.popsize_profiles,
        mutation=(0, 1.9),
        seed=n,
        polish=not pyargs.skip_polish_profiles,
        tol=pyargs.optim_tol_profiles,
        atol=pyargs.optim_atol_profiles,
        updating="immediate",
        recombination=0.3,
        init=init,
    )
    return res


def __calculate_profile_point_wrapper(args):
    return calculate_profile_point(*args)


def plot_profile(
    n: int,
    p_samples,
    costs,
    popt,
    final_cost: float,
    x0_bounds: dict,
    ax,
    color,
    label: str,
    displacement_error: float,
):
    # Filter out results that have produced errors
    filt = costs != ERROR_COST
    costs = costs[filt]
    p_samples = p_samples[filt]

    p_samples = np.array([*p_samples, popt[n]])
    costs = np.array([*costs, final_cost])
    ind = np.argsort(p_samples)
    p_samples = p_samples[ind]
    costs = costs[ind]

    d = displacement_error
    ax.plot(p_samples, (costs - final_cost) / d**2, color=color, label=label)
    ax.scatter(
        popt[n],
        0 * final_cost / d,
        marker="x",
        color=color,
        alpha=0.7,
        s=12**2,
    )
    name = list(x0_bounds.keys())[n].replace("_", " ")
    units = list(x0_bounds.items())[n][1][3]
    ax.set_xlabel(f"{name} {units}")
    ax.set_ylabel("PL(θ) - L(θ)")


def create_default_parameters(positions_data, iterations_data):
    parameters = generate_parameters()
    parameters.n_vertices = positions_data.shape[1]

    # Define size of the domain
    # Image has 604 pixels and 100 pixles correspond to 10µm
    parameters.domain_size = 604 / PIXELS_PER_MICRON  # in µm
    parameters.block_size = 200 / PIXELS_PER_MICRON  # in µm

    segments_data = np.linalg.norm(
        positions_data[:, 1:] - positions_data[:, :-1], axis=2
    )
    lengths_data = np.sum(segments_data, axis=1)

    # Set the initial rod length
    parameters.rod_length = np.sum(segments_data[0])  # in µm

    # Estimate the growth rate
    estimated_growth_rate = (
        np.log(lengths_data[1] / lengths_data[0] + 1) / parameters.t_max
    )
    parameters.growth_rate = estimated_growth_rate  # 1 /seconds

    parameters.dt = 0.01  # seconds
    parameters.t_max = iterations_data[1] - iterations_data[0]  # seconds
    parameters.save_interval = parameters.t_max  # seconds

    t_relax = iterations_data[2] - iterations_data[1]
    return parameters, t_relax


def compare_with_data(
    x0_bounds,
    positions_data,
    iterations_data,
    pyargs,
    set_params={},
    seed: int = 20,
    output_dir="out/crm_amir/result-full/",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdata = np.array(positions_data)

    for n, p in enumerate(positions_data):
        ind = np.argsort(p[:, 0])
        pdata[n] = p[ind] / PIXELS_PER_MICRON

    # x0 = [x[1] for _, x in x0_bounds.items()]
    bounds = [(x[0], x[2]) for _, x in x0_bounds.items()]
    try:
        pall = np.loadtxt(output_dir / "popt.csv")
        pfin = pall[0]
        popt = pall[1:]
        assert len(pall) == len(x0_bounds) + 1
        print(f"Loaded popt from {output_dir / 'popt.csv'}")
    except:
        res = sp.optimize.differential_evolution(
            objective_function,
            # x0,
            args=(set_params, pdata, iterations_data, x0_bounds, False, True),
            # method="L-BFGS-B",
            bounds=bounds,
            maxiter=pyargs.maxiter,
            popsize=pyargs.popsize,
            workers=pyargs.workers,
            tol=0,
            polish=not pyargs.skip_polish,
            mutation=(0, 1.9),
            seed=seed,
            updating="immediate",
            recombination=0.3,
            init=pyargs.init,
        )
        pfin = res.fun
        popt = res.x
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(output_dir / "popt.csv", np.array([pfin, *popt]))

    plot_results(popt, pdata, iterations_data, x0_bounds, set_params, output_dir)

    try:
        costs = np.loadtxt(output_dir / "profiles.csv")
        assert costs.shape == (pyargs.samples_profiles, len(popt))
        samples = np.loadtxt(output_dir / "samples.csv")
        print(f"Loaded Costs from {output_dir / 'profiles.csv'}")
    except:
        b_lower = [x[0] for x in x0_bounds.values()]
        b_upper = [x[2] for x in x0_bounds.values()]
        samples = np.linspace(b_lower, b_upper, pyargs.samples_profiles, endpoint=True)
        counts = np.array([np.arange(samples.shape[1])] * samples.shape[0])

        arglist = zip(
            counts.reshape(-1),
            samples.reshape(-1),
            itertools.repeat(popt),
            itertools.repeat(pdata),
            itertools.repeat(iterations_data),
            itertools.repeat(x0_bounds),
            itertools.repeat(set_params),
            itertools.repeat(pyargs),
        )

        pool = mp.Pool(pyargs.workers)
        results = tqdm(
            pool.imap(__calculate_profile_point_wrapper, arglist),
            total=int(np.prod(samples.shape)),
            desc="Plotting Profiles",
        )
        costs = np.array([r.fun for r in results]).reshape(counts.shape)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(output_dir / "samples.csv", samples)
        np.savetxt(output_dir / "profiles.csv", costs)

    params = {k: popt[n] for n, (k, _) in enumerate(x0_bounds.items())}
    return pfin, popt, costs, samples, params


def __render_single_snapshot(iter, agent, parameters, render_settings, output_dir):
    green = (np.uint8(44), np.uint8(189), np.uint8(25))
    agent.pos = agent.pos[:, [0, 2, 1]]
    cells = {crm.CellIdentifier.new_initial(0): (agent, None)}
    img = crm.imaging.render_pv_image(
        cells,
        render_settings,
        (parameters.domain_size, parameters.domain_size),
        colors={crm.CellIdentifier.new_initial(0): green},
    )
    block_size = np.round(
        parameters.block_size / parameters.domain_size * img.shape[1]
    ).astype(int)
    bg_filt = img == render_settings.bg_brightness
    img[:, :block_size][bg_filt[:, :block_size]] = int(
        render_settings.bg_brightness / 2
    )
    cv.imwrite(output_dir / f"{iter:010}.png", np.swapaxes(img, 0, 1)[::-1])


def render_snapshots(agents, parameters, output_dir):
    render_settings = crm.RenderSettings()
    render_settings.bg_brightness = 200

    for n, agent in tqdm(
        enumerate(agents), total=len(agents), desc="Rendering Snapshots"
    ):
        __render_single_snapshot(
            n,
            agent,
            parameters,
            render_settings,
            output_dir,
        )


def obtain_data(output_dir, n_vertices):
    output_dir = Path(output_dir)
    # data_files = glob("data/crm_amir/elastic/positions/*.txt")
    data_files = [
        (10, "data/crm_amir/elastic-segmented/000024-masked.png"),
        (17, "data/crm_amir/elastic-segmented/000032-masked.png"),
        (34, "data/crm_amir/elastic-segmented/000064-masked.png"),
    ]

    positions_data = np.array(
        [
            extract_mask(iter, cv.imread(df), n_vertices, output_dir)
            for iter, df in data_files
        ]
    )
    iterations_data = np.array([d[0] for d in data_files])
    iterations_data -= np.min(iterations_data)

    return positions_data, iterations_data


def crm_amir_main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-w", "--workers", type=int, default=-1, help="Number of threads"
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=350,
        help="Maximum iterations of the optimization routine",
    )
    parser.add_argument(
        "--popsize",
        type=int,
        default=30,
        help="Population Size of the optimization routine",
    )
    parser.add_argument(
        "--skip-polish",
        default=False,
        action="store_true",
        help="Skips polishing the result of the differential evolution algorithm",
    )
    parser.add_argument(
        "--maxiter-profiles",
        type=int,
        default=350,
        help="See MAXITER",
    )
    parser.add_argument(
        "--optim-tol-profiles",
        type=float,
        default=1e-4,
        help="Relative Tolerance for optimization within profiles",
    )
    parser.add_argument(
        "--optim-atol-profiles",
        type=float,
        default=1e-2,
        help="Absolute Tolerance for optimization within profiles",
    )
    parser.add_argument(
        "--init",
        type=str,
        default="latinhypercube",
        help="Initialization method for the sampling of parameters.",
    )
    parser.add_argument(
        "--popsize-profiles",
        type=int,
        default=30,
        help="See POPSIZE",
    )
    parser.add_argument(
        "--skip-polish-profiles",
        default=False,
        action="store_false",
        help="See POLISH",
    )
    parser.add_argument(
        "--samples-profiles",
        type=int,
        default=100,
        help="Number of sample points for profile likelihood plots",
    )
    pyargs = parser.parse_args()
    if pyargs.workers == -1:
        pyargs.workers = mp.cpu_count()

    crm.plotting.set_mpl_rc_params()


    positions_data, iterations_data = obtain_data("out/crm_amir/", n_vertices=20)

    # Define globals for all optimizations
    rod_rigidity = (50, 120.0, 400, "[µm/s²]")
    drag_force = (0.00003, 0.0001, 0.0003, "[1/s²µm²]")
    damping = (0.000, 0.1, 0.3, "[1/s]")
    growth_rate = (0.002, 0.01, 0.02, "[1/s]")
    spring_tension = (10, 30.0, 160.0, "[1/s²]")

    # Define which parameters should be optimized
    x0_bounds = {
        "rod_rigidity": rod_rigidity,
        "drag_force": drag_force,
        "damping": damping,
        "growth_rate": growth_rate,
        "spring_tension": spring_tension,
    }
    pfin1, popt1, costs1, samples1, params1 = compare_with_data(
        x0_bounds,
        positions_data,
        iterations_data,
        pyargs,
        output_dir="out/crm_amir/result-1",
    )

    x0_bounds_reduced = {
        "rod_rigidity": rod_rigidity,
        "drag_force": drag_force,
        # "damping": damping,
        "growth_rate": growth_rate,
        "spring_tension": spring_tension,
    }
    set_params = {"damping": 0}
    pfin2, popt2, costs2, samples2, params2 = compare_with_data(
        x0_bounds_reduced,
        positions_data,
        iterations_data,
        pyargs,
        set_params=set_params,
        output_dir="out/crm_amir/result-2/",
    )

    x0_bounds_3 = {
        "rod_rigidity": rod_rigidity,
        "drag_force": drag_force,
        # "damping": damping,
        "growth_rate": growth_rate,
        # "spring_tension": spring_tension,
    }
    # set_params_3 = {"spring_tension": params2["spring_tension"]}
    set_params_3 = {
        "spring_tension": params2["spring_tension"],
        "damping": 0,
    }
    pfin3, popt3, costs3, samples3, params3 = compare_with_data(
        x0_bounds_3,
        positions_data,
        iterations_data,
        pyargs,
        set_params=set_params_3,
        output_dir="out/crm_amir/result-3/",
    )

    # plot_angles_and_endpoints()
    displacement_error = 0.8

    output_dir = Path("out/crm_amir/profiles/")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))

    for n, name in enumerate(list(params1.keys())):
        crm.configure_ax(ax)
        ax.grid(False, which="minor")
        plot_profile(
            n,
            samples1[:, n],
            costs1[:, n],
            popt1,
            pfin1,
            x0_bounds,
            ax,
            color=COLOR2,
            label="Full",
            displacement_error=displacement_error,
        )

        names2 = np.array(list(params2.keys()))
        if name in names2:
            k = np.where(name == names2)[0][0]
            plot_profile(
                k,
                samples2[:, k],
                costs2[:, k],
                popt2,
                pfin2,
                x0_bounds_reduced,
                ax,
                color=COLOR3,
                label="$\\lambda=0$",
                displacement_error=displacement_error,
            )

        names3 = np.array(list(params3.keys()))
        if name in names3:
            m = np.where(name == names3)[0][0]
            plot_profile(
                m,
                samples3[:, m],
                costs3[:, m],
                popt3,
                pfin3,
                x0_bounds_3,
                ax,
                color=COLOR5,
                label="$\\gamma=\\gamma_\\text{opt}$",
                displacement_error=displacement_error,
            )

        all_costs = [costs1[:, n] - pfin1]
        if name in names2:
            all_costs.append(costs2[:, k] - pfin2)
        if name in names3:
            all_costs.append(costs3[:, m] - pfin3)
        min_costs = np.min(all_costs, axis=0) / displacement_error**2
        p_samples = samples1[:, n]  # This should be identical for all profiles

        # Fill confidence levels
        thresh_prev = 0
        for i, q in enumerate([0.68, 0.90, 0.95]):
            thresh = sp.stats.chi2.ppf(q, 1)
            color = crm.plotting.COLOR3 if i % 2 == 0 else crm.plotting.COLOR5
            y = min_costs
            filt = y <= thresh
            lower = np.max(np.array([y, np.repeat(thresh_prev, len(y))]), axis=0)
            ax.fill_between(
                p_samples,
                lower,
                np.repeat(thresh, len(lower)),
                where=filt,
                interpolate=True,
                color=color,
                alpha=0.3,
            )
            thresh_prev = thresh

        upper = np.min(
            [4 * thresh_prev, 1.05 * np.max([np.max(all_costs), thresh_prev])]
        )
        lower = -0.05 * upper
        ax.set_ylim(lower, upper)

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.10),
            ncol=3,
            frameon=False,
        )

        fig.savefig(output_dir / f"{name}.png")
        fig.savefig(output_dir / f"{name}.pdf")
        ax.cla()

    crm.configure_ax(ax, minor=False)
