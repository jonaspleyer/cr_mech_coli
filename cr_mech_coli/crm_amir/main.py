from cr_mech_coli.crm_amir import run_sim, Parameters
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

GREEN_COLOR = np.array([21.5 / 100, 86.6 / 100, 21.6 / 100]) * 255

ERROR_COST = 1e6
PIXELS_PER_MICRON = 102 / 10


def calculate_angle(p: np.ndarray, parameters: Parameters) -> float:
    intersection = np.array([parameters.block_size, parameters.domain_size / 2.0])
    endpoint = p[-1] if p[-1, 0] >= p[0, 0] else p[0]
    if endpoint[0] < parameters.block_size:
        return np.nan
    l1 = np.linalg.norm(endpoint - intersection)
    segments = np.linalg.norm(p[1:] - p[:-1], axis=1)
    l2 = np.sum(segments) - parameters.block_size
    angle = np.acos(l1 / np.clip(l2, 0, np.inf))
    return angle


def generate_parameters() -> Parameters:
    parameters = Parameters()
    parameters.block_size = 25.0
    parameters.dt = 0.01
    parameters.t_max = 200
    parameters.domain_size = 400
    n_vertices = 20
    parameters.n_vertices = n_vertices
    parameters.growth_rate = 0.03 * 7 / (n_vertices - 1)
    parameters.rod_rigiditiy = 20.0 * n_vertices / 20
    parameters.save_interval = 1.0
    parameters.damping = 0.02
    parameters.spring_tension = 10.0
    parameters.drag_force = 0.03
    return parameters


def plot_angles_and_endpoints():
    parameters = generate_parameters()

    endpoints = []
    y_collection = []
    rod_rigidities = np.linspace(0.3, 30, 20, endpoint=True)
    for rod_rigiditiy in rod_rigidities:
        parameters.rod_rigiditiy = rod_rigiditiy
        agents = run_sim(parameters)
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


def extract_mask(iteration, img, n_vertices: int, output_dir=None):
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

    if output_dir is not None:
        cv.imwrite(output_dir / f"progression-{iteration:06}-1.png", img)
        cv.imwrite(output_dir / f"progression-{iteration:06}-2.png", img2)
        cv.imwrite(output_dir / f"progression-{iteration:06}-3.png", img3)
        cv.imwrite(output_dir / f"progression-{iteration:06}-4.png", img4)

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
    positions,
    x0_bounds,
    return_all=False,
    print_output=False,
):
    parameters = create_default_parameters(positions)
    for k, v in set_params.items():
        parameters.__setattr__(k, v)

    # Variable Parameters
    for name, value in zip(x0_bounds.keys(), params):
        parameters.__setattr__(name, value)

    try:
        rods = run_sim(parameters)
    except ValueError:
        if print_output:
            print(f"ERROR Returning {ERROR_COST}")
        return ERROR_COST

    # Get initial and final position of rod
    p0 = rods[0][1].agent.pos[:, np.array([0, 2])]
    p1 = rods[-1][1].agent.pos[:, np.array([0, 2])]

    p0[:, 1] = parameters.domain_size - p0[:, 1]
    p1[:, 1] = parameters.domain_size - p1[:, 1]

    # Shift such that start points align

    positions = np.array(positions)
    positions = np.array([parameters.domain_size, 0]) - np.array([1, -1]) * positions
    positions = positions[:, ::-1]
    for i in range(positions.shape[0]):
        positions[i, 0, 0] -= positions[i, 0, 0]
    x_shift_positions0 = calculate_x_shift(positions[0], parameters.block_size)
    x_shift_p0 = calculate_x_shift(p0, parameters.block_size)
    x_shift_diff = x_shift_positions0 - x_shift_p0
    positions[:, :, 1] -= x_shift_diff

    if return_all:
        return p0, p1, positions, parameters

    diff = p1 - positions[1]
    cost = np.linalg.norm(diff)

    if print_output:
        print(f"f(x)={cost:>7.4f}", end=" ")
        for name, p in zip(x0_bounds.keys(), params):
            print(f"{name}={p:.4f}", end=" ")
        print()

    return cost


def plot_results(popt, positions: np.ndarray, x0_bounds: dict, set_params):
    p0, p1, positions, parameters = objective_function(
        popt, set_params, positions, x0_bounds, return_all=True
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    crm.configure_ax(ax)
    ax.plot(p0[:, 1], p0[:, 0], color=crm.plotting.COLOR5, linestyle=":")
    ax.plot(p1[:, 1], p1[:, 0], color=crm.plotting.COLOR3, linestyle=":")
    ax.plot(
        positions[0, :, 1],
        positions[0, :, 0],
        color=crm.plotting.COLOR5,
        linestyle="--",
        alpha=0.5,
    )
    x_shift = calculate_x_shift(positions[0], parameters.block_size)
    ax.scatter(x_shift, parameters.block_size, marker="x", color=crm.plotting.COLOR4)
    ax.plot(
        positions[1, :, 1],
        positions[1, :, 0],
        color=crm.plotting.COLOR3,
        linestyle="--",
        alpha=0.5,
    )

    # Define limits for plot
    dx = parameters.domain_size / 4
    ax.set_xlim(dx, parameters.domain_size - dx)
    ax.set_ylim(0, parameters.domain_size - dx)
    ax.fill_between(
        [0, parameters.domain_size],
        [parameters.block_size] * 2,
        color="gray",
        alpha=0.4,
    )
    ax.set_xlabel("[µm]")
    ax.set_ylabel("[µm]")
    fig.savefig("out/crm_amir/fit-comparison.png")
    fig.savefig("out/crm_amir/fit-comparison.pdf")
    plt.close(fig)


def calculate_profile_point(
    n: int,
    pnew: float,
    popt,
    positions: np.ndarray,
    x0_bounds: dict,
    set_params,
):
    x0_bounds_new = {k: v for i, (k, v) in enumerate(x0_bounds.items()) if i != n}
    x0 = [p for i, p in enumerate(popt) if i != n]
    bounds = [(x[0], x[2]) for x in x0_bounds_new.values()]

    assert len(x0_bounds_new) + 1 == len(x0_bounds)
    assert len(x0) == len(x0_bounds_new)
    assert len(x0) == len(bounds)
    # new_args = (args[0], copy.deepcopy(args[1]), x0_bounds_new)
    # parameters = create_default_parameters(positions)
    # parameters.__setattr__(list(x0_bounds.keys())[n], pnew)

    res = sp.optimize.differential_evolution(
        objective_function,
        # x0=x0,
        args=(set_params | {list(x0_bounds.keys())[n]: pnew}, positions, x0_bounds_new),
        # method="L-BFGS-B",
        bounds=bounds,
        maxiter=100,
        popsize=30,
        mutation=(0, 1.2),
        seed=n,
    )
    return res


def plot_profile(
    n: int,
    popt,
    final_cost: float,
    positions,
    x0_bounds: dict,
    workers: int,
    set_params,
    output_dir,
):
    b_lower = list(x0_bounds.values())[n][0]
    b_upper = list(x0_bounds.values())[n][2]
    p_samples = np.linspace(b_lower, b_upper)

    from itertools import repeat

    arglist = zip(
        repeat(n),
        p_samples,
        repeat(popt),
        repeat(positions),
        repeat(x0_bounds),
        repeat(set_params),
    )

    pool = mp.Pool(workers)
    results = pool.starmap(calculate_profile_point, arglist)
    # results = [calculate_profile_point(*r) for r in arglist]
    costs = np.array([r.fun for r in results])

    # Filter out results that have produced errors
    filt = costs != ERROR_COST
    costs = costs[filt]
    p_samples = p_samples[filt]

    p_samples = np.array([*p_samples, popt[n]])
    costs = np.array([*costs, final_cost])
    ind = np.argsort(p_samples)
    p_samples = p_samples[ind]
    costs = costs[ind]

    fig, ax = plt.subplots(figsize=(8, 8))
    crm.configure_ax(ax)

    ax.plot(p_samples, costs, color=crm.plotting.COLOR3)
    ax.scatter(popt[n], final_cost, marker="x", color="red", alpha=0.7)
    name = list(x0_bounds.keys())[n].replace("_", " ")
    ax.set_xlabel(name)
    ax.set_ylabel("Cost Function")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_dir / f"{name}.png")
    fig.savefig(output_dir / f"{name}.pdf")
    plt.close(fig)


def create_default_parameters(positions):
    parameters = generate_parameters()
    parameters.n_vertices = positions.shape[1]

    # Define size of the domain
    # Image has 604 pixels and 100 pixles correspond to 10µm
    parameters.domain_size = 604 / PIXELS_PER_MICRON  # in µm
    parameters.block_size = 200 / PIXELS_PER_MICRON  # in µm

    segments_data = np.linalg.norm(positions[:, 1:] - positions[:, :-1], axis=2)
    lengths_data = np.sum(segments_data, axis=1)

    # Set the initial rod length
    parameters.rod_length = np.sum(segments_data[0])  # in µm

    # Estimate the growth rate
    estimated_growth_rate = (
        np.log(lengths_data[1] / lengths_data[0] + 1) / parameters.t_max
    )
    parameters.growth_rate = estimated_growth_rate

    parameters.dt = 0.01
    parameters.t_max = 3
    parameters.save_interval = parameters.t_max

    return parameters


def compare_with_data(
    x0_bounds,
    workers: int,
    set_params={},
    n_vertices: int = 20,
    output_dir="out/crm_amir/profiles-full/",
):
    # data_files = glob("data/crm_amir/elastic/positions/*.txt")
    data_files = [
        (24, "data/crm_amir/elastic/frames/000024.png"),
        (32, "data/crm_amir/elastic/frames/000032.png"),
    ]

    positions = np.array(
        [extract_mask(iter, cv.imread(df), n_vertices) for iter, df in data_files]
    )

    for n, p in enumerate(positions):
        ind = np.argsort(p[:, 0])
        positions[n] = p[ind] / PIXELS_PER_MICRON

    # x0 = [x[1] for _, x in x0_bounds.items()]
    bounds = [(x[0], x[2]) for _, x in x0_bounds.items()]
    res = sp.optimize.differential_evolution(
        objective_function,
        # x0,
        args=(set_params, positions, x0_bounds, False, True),
        # method="L-BFGS-B",
        bounds=bounds,
        maxiter=200,
        popsize=30,
        workers=workers,
        tol=0,
        polish=True,
        mutation=(0, 1.2),
        seed=n_vertices,
    )

    plot_results(res.x, positions, x0_bounds, set_params)

    for n in tqdm(
        range(len(x0_bounds)), total=len(x0_bounds), desc="Plotting Profiles"
    ):
        plot_profile(
            n, res.x, res.fun, positions, x0_bounds, workers, set_params, output_dir
        )


def __render_single_snapshot(iter, agent, parameters, render_settings):
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
    cv.imwrite(f"out/crm_amir/{iter:010}.png", np.swapaxes(img, 0, 1)[::-1])


def render_snapshots():
    parameters = generate_parameters()
    agents = run_sim(parameters)

    n_saves = 10

    save_points = np.clip(
        np.round(np.linspace(0, len(agents), n_saves)).astype(int), 0, len(agents) - 1
    )

    render_settings = crm.RenderSettings()
    render_settings.bg_brightness = 200

    for save_point in tqdm(
        save_points, total=len(save_points), desc="Rendering Snapshots"
    ):
        __render_single_snapshot(
            save_point, agents[save_point][1].agent, parameters, render_settings
        )


def crm_amir_main():
    crm.plotting.set_mpl_rc_params()
    # render_snapshots()

    # Define which parameters should be optimized
    x0_bounds = {
        "rod_rigiditiy": (0.0001, 20.0, 250.0),  # rod_rigiditiy,
        "drag_force": (0.0000, 0.1, 10.0),  # drag_force,
        "damping": (0.000, 1.0, 2.0),  # damping,
        "growth_rate": (0.0, 0.01, 2.0),  # growth_rate,
        "spring_tension": (0.0000, 0.01, 30.0),  # spring_tension
    }
    compare_with_data(x0_bounds, workers=30)

    x0_bounds_reduced = {
        "rod_rigiditiy": (0.0001, 20.0, 250.0),  # rod_rigiditiy,
        "drag_force": (0.0000, 0.1, 10.0),  # drag_force,
        # "damping": (0.000, 1.0, 2.0),  # damping,
        "growth_rate": (0.0, 0.01, 2.0),  # growth_rate,
        # "spring_tension": (0.0000, 0.01, 30.0),  # spring_tension
    }
    set_params = {
        "damping": 0.0,
        "spring_tension": 20,
    }
    compare_with_data(
        x0_bounds_reduced,
        workers=30,
        set_params=set_params,
        output_dir="out/crm_amir/profiles-reduced/",
    )
    # plot_angles_and_endpoints()
