import cr_mech_coli as crm
import PIL
import numpy as np
import matplotlib as mpl


if __name__ == "__main__":
    grs = crm.GrowthRateSetter({"mean": 0.01, "std": 0.0015})
    slts = crm.SpringLengthThresholdSetter({"mean": 8.0, "std": 1.5})
    agent_settings = crm.AgentSettings(
        growth_rate=grs.mean,
        spring_length_threshold=slts.mean,
        damping=0.025,
    )
    agent_settings.growth_rate_setter = grs
    agent_settings.spring_length_threshold_setter = slts

    agent_settings.interaction.potential_stiffness = 0.8
    agent_settings.interaction.strength = 0.3
    agent_settings.interaction.cutoff = 2.0 * agent_settings.interaction.radius

    config = crm.Configuration()
    config.t0 = 0.0
    config.dt = 0.1
    config.t_max = 750.0
    config.n_saves = 1
    config.domain_size = (700, 700)
    config.n_threads = 14
    config.n_voxels = (14, 14)

    config.surface_friction = 0
    config.gel_pressure = 0
    config.domain_height = 1e-1

    def render_img(seed):
        config.progressbar = f"Big Sim seed={seed}"
        positions = crm.generate_positions(
            n_agents=4,
            agent_settings=agent_settings,
            config=config,
            rng_seed=seed,
            dx=(config.domain_size[0] * 0.4, config.domain_size[1] * 0.4),
            n_vertices=6,
        )
        for pos in positions:
            pos[:, 2] = config.domain_height / 2

        agent_dict = agent_settings.to_rod_agent_dict()
        agents = [crm.RodAgent(p, 0.0 * p, **agent_dict) for p in positions]

        sim_result = crm.run_simulation_with_agents(config, agents)
        render_settings = crm.RenderSettings()
        render_settings.pixel_per_micron = 1

        last_iter = sim_result.get_all_iterations()[-1]

        cell_to_color = sim_result.cell_to_color

        cmap = mpl.colormaps["twilight"]

        # Assign color depending on alignment
        for c in sim_result.cells[last_iter]:
            cell = sim_result.cells[last_iter][c][0]
            pos = cell.pos
            q = pos[1:] - pos[:-1]
            angle = np.mean([np.arctan2(x[1], x[0]) for x in q])
            angle = angle % np.pi
            new_color = np.array(cmap(angle / np.pi))[:3] * 255
            cell_to_color[c] = (int(new_color[0]), int(new_color[1]), int(new_color[2]))

        img = crm.render_approximate_mask(
            sim_result.cells[last_iter],
            cell_to_color,
            (config.domain_size[0], config.domain_size[1]),
            resolution=(int(config.domain_size[0] * 4), int(config.domain_size[1] * 4)),
            epsilon=0.1,
        )
        img = PIL.Image.fromarray(img)
        img.save("docs/source/_static/big-sim.png")

    render_img(3)
