import cr_mech_coli as crm
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def run_sim(ml_config, randomize_positions=0.02, n_vertices=8) -> crm.CellContainer:
    positions = np.array(
        crm.generate_positions_old(
            n_agents=1,
            agent_settings=ml_config.agent_settings,
            config=ml_config.config,
            rng_seed=ml_config.rng_seed,
            dx=ml_config.dx,
            randomize_positions=randomize_positions,
            n_vertices=n_vertices,
        )
    )
    positions[:, :, 2] = 0.1 * ml_config.agent_settings.interaction.radius
    agent_dict = ml_config.agent_settings.to_rod_agent_dict()

    agents = [crm.RodAgent(p, 0.0 * p, **agent_dict) for p in positions]

    return crm.run_simulation_with_agents(ml_config.config, agents)


def produce_ydata(container):
    cells = container.get_cells()
    iterations = container.get_all_iterations()
    positions = [np.array([c[0].pos for c in cells[i].values()]) for i in iterations]
    y = [np.max(p[:, :, 2]) for p in positions]
    return iterations, positions, y


def crm_multilayer_main():
    # Create many Multilayer-Configs
    ml_config = crm.crm_multilayer.MultilayerConfig()
    ml_config.config.dt = 0.05
    ml_config.config.n_saves = 20
    ml_config.config.t_max = 300
    ml_config.config.domain_height = 20.0
    ml_config.config.domain_size = (400, 400)
    ml_config.dx = (180, 180)
    ml_config.config.n_voxels = (10, 10)
    ml_config.config.gravity = 0.1
    ml_config.config.show_progressbar = True

    ml_config.agent_settings.mechanics.damping = 0.1
    ml_config.agent_settings.mechanics.rigidity = 15
    ml_config.agent_settings.interaction.strength = 0.2

    ml_configs = [ml_config.clone_with_args(rng_seed=seed) for seed in range(1)]
    # Produce data for various configs

    iterations = []
    y = []
    n_agents = []
    for m in ml_configs:
        container = run_sim(m)
        i, positions, yi = produce_ydata(container)
        n_agents.append([p.shape[0] for p in positions])
        iterations.append(i)
        y.append(yi)

    y = np.array(y)
    iterations = np.array(iterations)

    t = np.mean(iterations, axis=0) * ml_config.config.dt
    yplt = np.mean(y, axis=0)
    yerr = np.std(y, axis=0)
    n_agents = np.array(n_agents)
    r = ml_config.agent_settings.interaction.radius * np.ones(t.shape[0])

    fig, ax = plt.subplots()
    ax.errorbar(t, yplt, yerr, label="Average y-coodinate", c="k")
    # ax.plot(t, 1 * r, label="Multiple Cell Radius $nR$", c="gray", linestyle="--")

    yticks = r[0] * np.arange(np.ceil(np.max(yplt) / r[0]))
    yticklabels = [f"${i}R$" for i, _ in enumerate(yticks)]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.grid(True, "major", axis="y")

    ax.set_ylabel("Height [µm]")
    ax.set_xlabel("Time [min]")
    ax.legend()

    ax2 = ax.twiny()
    ax2.set_xlabel("Number of Bacteria")
    n_ticks = len(ax.get_xticks())
    filt = np.round(np.linspace(0, len(t) - 1, n_ticks)).astype(int)
    labels = [x for x in np.mean(n_agents[:, filt], axis=0).astype(int)]
    labels[0] = None
    labels[-1] = None
    ax2.set_xticks(filt)
    ax2.set_xticklabels(labels)

    fig.savefig("tmp.png")
