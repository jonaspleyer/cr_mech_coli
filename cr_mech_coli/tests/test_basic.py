import cr_mech_coli as crm
from tqdm import tqdm

if __name__ == "__main__":
    config = crm.Configuration()
    config.show_progressbar = True
    config.n_agents = 8
    config.t0 = 0.0
    config.dt = 0.1
    config.t_max = 120.0
    config.save_interval = 10.0
    config.agent_settings.mechanics.spring_tension = 0.2
    config.agent_settings.mechanics.damping = 1.0
    config.agent_settings.mechanics.angle_stiffness = 0.5
    config.agent_settings.mechanics.spring_length = 3.0
    config.agent_settings.interaction.strength = 0.1
    config.agent_settings.interaction.potential_stiffness = 0.5
    config.agent_settings.interaction.radius = 3.0
    config.agent_settings.interaction.cutoff = 10.0
    config.agent_settings.growth_rate = 20.0
    config.agent_settings.spring_length_threshold = 6.0

    try:
        cells = crm.run_simulation(config)
        colors = crm.assign_colors_to_cells(cells)
        iterations = sorted(cells.keys())
        render_settings = crm.RenderSettings()
        render_settings.noise = 50
        render_settings.kernel_size = 30
        render_settings.ssao_radius = 50

        for iteration in tqdm(iterations, total=len(iterations)):
            crm.render_image(
                config,
                cells[iteration],
                render_settings,
                filename="out/images/snapshot_{:07}.png".format(iteration),
            )
            crm.render_mask(
                config,
                cells[iteration],
                colors,
                render_settings,
                filename="out/masks/snapshot_{:07}-mask.png".format(iteration),
            )
    except Exception as error:
        print(error)
