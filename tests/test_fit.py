import cr_mech_coli as crm
import numpy as np


def produce_masks():
    config = crm.Configuration()
    config.t0 = 0.0
    config.dt = 0.1
    config.t_max = 100.0
    config.n_saves = 4
    agent_settings = crm.AgentSettings(growth_rate=0.05)
    positions = crm.generate_positions_old(
        4,
        agent_settings,
        config,
        rng_seed=11,
    )
    agents = [
        crm.RodAgent(pos=p, vel=0 * p, **agent_settings.to_rod_agent_dict())
        for p in positions
    ]

    cell_container = crm.run_simulation_with_agents(config, agents)

    all_cells = cell_container.get_cells()
    iterations = cell_container.get_all_iterations()
    colors = cell_container.cell_to_color
    i1 = iterations[1]
    i2 = iterations[2]

    domain_size = config.domain_size
    rs = crm.RenderSettings()
    mask1 = crm.render_mask(all_cells[i1], colors, domain_size, render_settings=rs)
    mask2 = crm.render_mask(all_cells[i2], colors, domain_size, render_settings=rs)
    return mask1, mask2, cell_container


def test_area_diff():
    mask1, mask2, _ = produce_masks()

    p1 = crm.penalty_area_diff(mask1, mask2)
    p2 = crm.penalty_area_diff(mask1, mask1)
    p3 = crm.penalty_area_diff(mask2, mask2)

    assert p1 > 0
    assert p2 == 0
    assert p3 == 0

    assert p1 <= 1
    assert p2 <= 1
    assert p3 <= 1


def test_area_diff_parents():
    mask1, mask2, cell_container = produce_masks()

    p1 = crm.penalty_area_diff_account_parents(mask1, mask2, cell_container)
    p2 = crm.penalty_area_diff_account_parents(mask1, mask1, cell_container)
    p3 = crm.penalty_area_diff_account_parents(mask2, mask2, cell_container)

    assert p1 > 0
    assert p2 == 0
    assert p3 == 0

    assert p1 <= 1
    assert p2 <= 1
    assert p3 <= 1


def test_area_diff_comparison():
    mask1, mask2, cell_container = produce_masks()

    q1 = crm.penalty_area_diff(mask1, mask2)
    p1 = crm.penalty_area_diff_account_parents(mask1, mask2, cell_container)

    assert p1 < q1


def test_area_diff_with_mask():
    mask1, mask2, _ = produce_masks()

    m = crm.area_diff_mask(mask1, mask2)
    p1 = np.mean(m)
    p2 = crm.penalty_area_diff(mask1, mask2)

    assert np.abs(p1 - p2) < 1e-4
