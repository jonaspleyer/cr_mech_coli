from cr_mech_coli.crm_amir import run_sim, Parameters
import cr_mech_coli as crm
import numpy as np
import cv2 as cv
from glob import glob
import matplotlib.pyplot as plt
import matplotlib as mpl


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
    parameters.rod_rigiditiy = 4.0
    parameters.block_size = 25.0
    parameters.dt = 0.01
    parameters.t_max = 200
    parameters.domain_size = 400
    n_vertices = 20
    parameters.n_vertices = n_vertices
    parameters.growth_rate = 0.02 * 7 / (n_vertices - 1)
    parameters.rod_rigiditiy = 2.0 * n_vertices / 20
    parameters.save_interval = 1.0
    parameters.damping = 1.0
    parameters.spring_tension = 10.0
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
    crm.plotting.set_mpl_rc_params()
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


def compare_with_data():
    data_files = glob("data/crm_amir/elastic/positions/*.txt")
    positions = np.array([np.genfromtxt(data_file) for data_file in data_files])

    # p = agents[-1][1].agent.pos[:, [0, 2]]

    # TODO this pixel size should not be given explicitly but rather read out
    # pos = crm.convert_cell_pos_to_pixels(
    #     p, (parameters.domain_size, parameters.domain_size), (604, 638)
    # )


def render_snapshots():
    parameters = generate_parameters()
    agents = run_sim(parameters)

    n_saves = 10

    save_points = np.clip(
        np.round(np.linspace(0, len(agents), n_saves)).astype(int), 0, len(agents) - 1
    )

    render_settings = crm.RenderSettings()
    render_settings.bg_brightness = 200
    for sp in save_points:
        green = np.array([44 / 255, 189 / 255, 25 / 255])
        agent = agents[sp][1].agent
        agent.pos = agent.pos[:, [0, 2, 1]]
        cells = {(0, 0): (agent, None)}
        img = crm.imaging.render_pv_image(
            cells,
            render_settings,
            (parameters.domain_size, parameters.domain_size),
            colors={(0, 0): green},
        )
        block_size = np.round(
            parameters.block_size / parameters.domain_size * img.shape[1]
        ).astype(int)
        bg_filt = img == render_settings.bg_brightness
        img[:, :block_size][bg_filt[:, :block_size]] = int(
            render_settings.bg_brightness / 2
        )
        cv.imwrite(f"out/crm_amir/{sp:010}.png", np.swapaxes(img, 0, 1)[::-1])


def crm_amir_main():
    # render_snapshots()
    # compare_with_data()
    plot_angles_and_endpoints()
