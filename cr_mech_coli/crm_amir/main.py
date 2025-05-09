from cr_mech_coli.crm_amir import run_sim, Parameters
import matplotlib.pyplot as plt
import numpy as np


def crm_amir_main():
    parameters = Parameters()
    parameters.rod_rigiditiy = 4.0
    parameters.dt = 0.01
    agents = run_sim(parameters)

    fig, ax = plt.subplots(figsize=(8, 8))

    n_saves = 12
    for i in np.linspace(0, len(agents), n_saves):
        i = np.clip(np.round(i).astype(int), 0, len(agents) - 1)
        p = agents[i][1].agent.pos
        red = np.array([1.0, 0.1, 0.0])
        ax.plot(p[:, 0], p[:, 2], color=red * i / len(agents))

    ax.set_xlim(0, parameters.domain_size)
    ax.set_ylim(0, parameters.domain_size)

    plt.show()
