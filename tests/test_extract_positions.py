import cr_mech_coli as crm
import numpy as np


def gen_results():
    config = crm.Configuration()
    config.domain_size = (150, 100)
    agent_settings = crm.AgentSettings()
    positions = crm.generate_positions(1, agent_settings, config)
    agent_settings.growth_rate = 0
    agents = [
        crm.RodAgent(pos=p, vel=p * 0.0, **agent_settings.to_rod_agent_dict())
        for p in positions
    ]

    cell_container = crm.run_simulation_with_agents(config, agents)
    return config, agents, cell_container


def gen_masks(config, cell_container: crm.CellContainer):
    iterations = cell_container.get_all_iterations()[-1:]
    all_cells = cell_container.get_cells()
    colors = cell_container.cell_to_color
    return [
        crm.render_mask(all_cells[it], colors, config.domain_size) for it in iterations
    ]


def test_extract_positions_1():
    config, agents, cell_container = gen_results()

    agent = agents[0]
    masks = gen_masks(config, cell_container)
    m2 = masks[-1]

    # Extraced position in units of pixels
    pe_pix = crm.extract_positions(m2)[0][0]

    # Exact position of agent
    pa_spa = agent.pos[:, :2]
    pa_pix = crm.convert_cell_pos_to_pixels(
        agent.pos[:, :2], config.domain_size, m2.shape[:2]
    )

    # Extraced position: Automatically converted to units of space
    pe_spa_conv_auto = crm.extract_positions(m2, domain_size=config.domain_size)[0][0]
    # Extraced position: Manually converted to units of space
    pe_spa_conv_manu = crm.convert_pixel_to_position(
        pe_pix, config.domain_size, m2.shape[:2]
    )

    for xy1, xy2 in zip(
        np.round(pe_pix).astype(int),
        np.round(pa_pix).astype(int),
    ):
        x1, y1 = np.min([xy1, m2.shape[:2]], axis=0)
        x2, y2 = np.min([xy2, m2.shape[:2]], axis=0)
        assert np.all(m2[x1, y1] != [0, 0, 0])
        assert np.all(m2[x2, y2] != [0, 0, 0])

    assert np.all(np.abs(pe_spa_conv_manu - pe_spa_conv_auto) < 1e-4)

    pa_pix_middle = np.mean(pa_pix, axis=0, dtype=np.float32)
    pe_pix_middle = np.mean(pe_pix, axis=0, dtype=np.float32)
    assert np.all(np.abs(pa_pix_middle - pe_pix_middle) < 1.0)

    pe_spa_conv_auto_middle = np.mean(pe_spa_conv_auto, axis=0)
    pa_spa_middle = np.mean(pa_spa, axis=0)
    assert np.all(np.abs(pe_spa_conv_auto_middle - pa_spa_middle) < 1)
