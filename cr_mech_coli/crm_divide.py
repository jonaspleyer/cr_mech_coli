import numpy as np
from glob import glob
from pathlib import Path
from tqdm import tqdm

import cr_mech_coli as crm
from cr_mech_coli import crm_fit

data_dir = Path("data/crm_fit/0002/")


def predict(
    initial_positions,
    settings,
    radius=3.0,
    strength=10.584545,
    bound=10,
    cutoff=100,
    en=0.21933548,
    em=0.50215733,
):
    # Define agents
    interaction = crm.MiePotentialF32(
        radius,
        strength,
        bound,
        cutoff,
        en,
        em,
    )
    agents = [
        crm.RodAgent(
            pos,
            vel=0 * pos,
            interaction=interaction,
            diffusion_constant=0.0,
            spring_tension=3.0,
            rigidity=10.0,
            spring_length=3.0,
            damping=2.5,
            growth_rate=0.00,
            growth_rate_distr=(0.00, 0.0),
            spring_length_threshold=12.0,
            neighbor_reduction=None,
        )
        for pos in initial_positions
    ]

    # define config
    config = settings.to_config()
    config.show_progressbar = True

    container = crm.run_simulation_with_agents(config, agents)
    pass


def main():
    files_images = sorted(glob(str(data_dir / "images/*")))
    files_masks = sorted(glob(str(data_dir / "masks/*.csv")))
    masks = [np.loadtxt(fm, delimiter=",") for fm in files_masks]

    settings = crm_fit.Settings.from_toml(data_dir / "settings.toml")
    n_vertices = settings.constants.n_vertices

    iterations_all = []
    positions_all = []
    lengths_all = []
    colors_all = []
    for mask, filename in tqdm(
        zip(masks, files_masks), total=len(masks), desc="Extract positions"
    ):
        try:
            pos, length, _, colors = crm.extract_positions(
                mask, n_vertices, domain_size=settings.constants.domain_size
            )
            positions_all.append(pos)
            lengths_all.append(length)
            iterations_all.append(int(Path(filename).stem.split("-")[0]))
            colors_all.append(colors)
        except ValueError as e:
            print("Encountered Error during extraction of positions:")
            print(filename)
            print(e)
            print("Omitting this particular result.")

    positions_initial = positions_all[0]
    print(positions_initial.shape)
    domain_height = settings.domain_height
    positions_initial = np.append(
        positions_initial,
        domain_height / 2 + np.zeros((*positions_initial.shape[:2], 1)),
        axis=2,
    ).astype(np.float32)
    predict(positions_initial, settings)
