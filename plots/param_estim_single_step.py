from cv2 import imread
import cr_mech_coli as crm
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from tqdm import tqdm

def predict(
        # Parameters
        growth_rate: float,# Shape (N)
        radius: float,
        strength: float,
        cutoff: float,
        potential_stiffness: float,
        rigidity: float,
        # Constants
        positions: np.ndarray,# Shape (N, n_vertices, 3)
        domain_size: float,
        return_cells: bool = False,
    ):
    config = crm.Configuration(
        domain_size=domain_size,
    )
    config.dt *= 0.25
    n_agents = positions.shape[0]
    n_vertices = positions.shape[1]

    def pos_to_spring_length(pos):
        res = np.sum(np.linalg.norm(pos[1:]-pos[:-1], axis=1)) / n_vertices
        return res

    agents = [
        crm.RodAgent(
            pos=np.array(
                [*positions[i].T, [config.domain_height / 2]*positions.shape[1]],
                dtype=np.float32
            ).T,
            vel=np.zeros((positions.shape[1], positions.shape[2]+1), dtype=np.float32),
            growth_rate=growth_rate,
            radius=radius,
            spring_length=pos_to_spring_length(positions[i]),
            spring_length_threshold=1000,
            strength=strength,
            potential_stiffness=potential_stiffness,
            cutoff=cutoff,
            rigidity=rigidity,
        )
        for i in range(n_agents)
    ]
    cell_container = crm.run_simulation_with_agents(config, agents)
    iterations = cell_container.get_all_iterations()
    agents_predicted = cell_container.get_cells_at_iteration(iterations[-1])
    if return_cells:
        return cell_container
    else:
        return np.array([agent.pos for agent, _ in agents_predicted.values()])

def predict_flatten(
        parameters,
        n_agents,
        cutoff,
        domain_size,
        pos_initial,
        pos_final,
        return_cells: bool = False,
    ):
    (growth_rate, radius, strength, potential_stiffness, rigidity) = parameters
    pos_predicted = predict(
        growth_rate,
        radius,
        strength,
        cutoff,
        potential_stiffness,
        rigidity,
        pos_initial,
        domain_size,
        return_cells=return_cells,
    )
    if return_cells:
        return pos_predicted
    # TODO
    # This is currently very inefficient.
    # We could probably better match the
    # positions to each other
    cost = np.min([min(
        np.sum((pos_predicted[i][:,:2] - pos_final[j])**2),
        np.sum((pos_predicted[i][:,:2] - pos_final[j])**2),
        )
    for j in range(len(pos_final)) for i in range(len(pos_predicted))])
    return cost

if __name__ == "__main__":
    # markers = np.fromfile("./data/growth-2-marked/image001042-markers.tif").reshape(576, 768)
    mask1 = np.loadtxt("data/growth-2-marked/image001042-markers.csv", delimiter=",")
    img1 = imread("data/growth-2/image001042.png")
    mask2 = np.loadtxt("data/growth-2-marked/image001052-markers.csv", delimiter=",")
    img2 = imread("data/growth-2/image001052.png")
    n_vertices = 8
    pos1 = np.array(crm.extract_positions(mask1, n_vertices))
    pos2 = np.array(crm.extract_positions(mask2, n_vertices))

    fig, ax = plt.subplots(3, 2, figsize=(12, 12))
    ax[0, 0].imshow(img1)
    ax[0, 1].imshow(img2)
    ax[1, 0].imshow(mask1)
    ax[1, 1].imshow(mask2)

    domain_size = np.max(mask1.shape)
    n_agents = len(pos1)
    cutoff=30.0
    args=(n_agents, cutoff, domain_size, pos1, pos2)

    growth_rate = 0.03
    radius = 8.0
    strength = 0.1
    potential_stiffness = 0.2
    rigidity = 0.8
    parameters = (growth_rate, radius, strength, potential_stiffness, rigidity)

    # Optimize values
    bounds = [
        [0.00, 0.05],# Growth Rate
        [4.0, 10.0],# Radius
        [0.1, 0.4],# Strength
        [0.2, 0.6],# Potential Stiffness
        [0.4, 1.0],# Rigidity
    ]
    res = sp.optimize.differential_evolution(
        predict_flatten,
        bounds=bounds,
        x0=parameters,
        args=args,
        workers=-1,
        updating='deferred',
        maxiter=20,
        disp=True,
        tol=1e-3,
        recombination=0.3,
        popsize=128,
    )

    param_infos = [
        ("Growth Rate", "\\mu m\\text{min}^{-1}"),
        ("Radius", "\\mu m"),
        ("Strength", "\\mu m^2\text{min}^{-2}"),
        ("Potential Stiffness", "\\mu m"),
        ("Rigidity", "\\mu m\\text{min}^{-1}"),
    ]

    # Plot Cost function against varying parameters
    for n, (p, bound) in enumerate(zip(res.x, bounds)):
        fig2, ax2 = plt.subplots()
    
        x = np.linspace(bound[0], bound[1], 20)
        ps = [
            [pi if n!=i else xi for i, pi in enumerate(res.x)]
            for xi in x
        ]
        y = [predict_flatten(p, *args) for p in ps]

        (name, units) = param_infos[n]

        ax2.set_title(name)
        ax2.set_ylabel("Cost function $L$")
        ax2.set_xlabel("Parameter Value [${}$]".format(units))
        ax2.scatter(p, res.fun, marker="o", color="red")
        ax2.plot(x, y)
        fig2.tight_layout()
        plt.savefig("docs/source/_static/fitting-methods/estimate-parameters1/{}.png".format(name))
        plt.close(fig2)

    cell_container = predict_flatten(
        res.x,
        *args,
        return_cells=True,
    )

    iterations = cell_container.get_all_iterations()
    agents_initial = cell_container.get_cells_at_iteration(iterations[0])
    agents_predicted = cell_container.get_cells_at_iteration(iterations[-1])

    mask_gen1 = crm.render_mask(agents_initial, cell_container.cell_to_color, domain_size)
    mask_gen2 = crm.render_mask(agents_predicted, cell_container.cell_to_color, domain_size)
    ax[2, 0].imshow(mask_gen1[::-1])
    ax[2, 1].imshow(mask_gen2[::-1])

    for p in pos1:
        ax[0, 0].plot(p[:,0], p[:,1], color="white")
    # for p in pos2:
    #     ax[0, 1].plot(p[:,0], p[:,1], color="white")
    for agent, _ in agents_predicted.values():
        p = agent.pos
        ax[0, 1].plot(p[:,0], p[:,1], color="white")
    fig.tight_layout()
    plt.show()
