import cr_mech_coli as crm
import PIL
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import string

MINUTE = 1
HOUR = 60 * MINUTE
DAY = 24 * HOUR
MICRON = 1

if __name__ == "__main__":
    grs = crm.GrowthRateSetter({"mean": 0.8 / HOUR, "std": 0.1 / HOUR})
    slts = crm.SpringLengthThresholdSetter({"mean": 0.9 * MICRON, "std": 0.2 * MICRON})
    agent_settings = crm.AgentSettings(
        growth_rate=grs.mean,
        spring_length=0.4 * MICRON,
        spring_length_threshold=slts.mean,
        damping=0.1 / MINUTE,
        spring_tension=10 / MINUTE,
        rigidity=1.1 / MINUTE,
    )
    agent_settings.growth_rate_setter = grs
    agent_settings.spring_length_threshold_setter = slts

    agent_settings.interaction.radius = 0.4 * MICRON
    agent_settings.interaction.potential_stiffness = 2.0 / MICRON
    agent_settings.interaction.strength = 1.0 * MICRON**2 / MINUTE**2
    agent_settings.interaction.cutoff = 2.0 * agent_settings.interaction.radius

    config = crm.Configuration()
    config.t0 = 0.0
    config.dt = 0.02 * MINUTE
    config.t_max = 9 * HOUR
    config.n_saves = 6
    config.domain_size = (70 * MICRON, 70 * MICRON)
    config.n_threads = 14
    config.n_voxels = (14, 14)

    config.surface_friction = 0
    config.gel_pressure = 0
    config.domain_height = 1e-1 * MICRON

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

        s = 0.01
        iterations = sim_result.get_all_iterations()
        fig, axs = plt.subplots(
            2,
            int(len(iterations) / 2),
            gridspec_kw={
                "left": 0,
                "right": 1,
                "bottom": 0,
                "top": 1,
                "wspace": s,
                "hspace": s,
            },
            figsize=(24, 12 - s / 2),
        )
        for ax, it, label in zip(axs.flatten(), iterations, string.ascii_uppercase):
            cell_to_color = sim_result.cell_to_color

            cmap = mpl.colormaps["twilight"]

            # Assign color depending on alignment
            print("N Cells: ", len(sim_result.cells[it]))
            for c in sim_result.cells[it]:
                cell = sim_result.cells[it][c][0]
                pos = cell.pos
                q = pos[1:] - pos[:-1]
                angle = np.mean([np.arctan2(x[1], x[0]) for x in q])
                angle = angle % np.pi
                new_color = np.array(cmap(angle / np.pi))[:3] * 255
                cell_to_color[c] = (
                    int(new_color[0]),
                    int(new_color[1]),
                    int(new_color[2]),
                )

            img = crm.render_approximate_mask(
                sim_result.cells[it],
                cell_to_color,
                (config.domain_size[0], config.domain_size[1]),
                resolution=(
                    int(config.domain_size[0] * 40),
                    int(config.domain_size[1] * 40),
                ),
                epsilon=1.0,
            )

            ax.imshow(img, aspect="equal")
            ax.set_axis_off()
            ax.text(
                0.05,
                0.95,
                label,
                fontsize=40,
                fontweight="semibold",
                fontfamily="serif",
                va="top",
                horizontalalignment="left",
                transform=ax.transAxes,
                color="white",
            )

            img = PIL.Image.fromarray(img)
            img.save(f"docs/source/_static/big-sim-{it:08}.png")

        fig.savefig("docs/source/_static/big-sim-series.png")

    render_img(7)
