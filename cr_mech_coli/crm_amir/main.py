from cr_mech_coli.crm_amir import run_sim, Parameters
import cr_mech_coli as crm
import numpy as np
import cv2 as cv


def crm_amir_main():
    parameters = Parameters()
    parameters.rod_rigiditiy = 4.0
    parameters.block_size = 50.0
    parameters.dt = 0.01
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
