import cr_mech_coli as crm
import numpy as np


def __create_config():
    config = crm.Configuration()
    config.t0 = 0.0
    config.dt = 0.1
    config.t_max = 100.0
    config.n_saves = 4
    config.domain_size = (config.domain_size[0], 1.25 * config.domain_size[1])
    return config


def produce_masks(growth_rate=0.05):
    config = __create_config()

    agent_settings = crm.AgentSettings(growth_rate=growth_rate)
    agents = crm.generate_agents(4, agent_settings, config, rng_seed=11)

    cell_container = crm.run_simulation_with_agents(config, agents)

    all_cells = cell_container.get_cells()
    iterations = cell_container.get_all_iterations()
    colors = cell_container.cell_to_color
    i1 = iterations[1]
    i2 = iterations[-1]

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

    p1 = crm.penalty_area_diff_account_parents(
        mask1, mask2, cell_container.color_to_cell, cell_container.parent_map
    )
    p2 = crm.penalty_area_diff_account_parents(
        mask1, mask1, cell_container.color_to_cell, cell_container.parent_map
    )
    p3 = crm.penalty_area_diff_account_parents(
        mask2, mask2, cell_container.color_to_cell, cell_container.parent_map
    )

    assert p1 > 0
    assert p2 == 0
    assert p3 == 0

    assert p1 <= 1
    assert p2 <= 1
    assert p3 <= 1


def test_area_diff_comparison():
    mask1, mask2, cell_container = produce_masks()

    q1 = crm.penalty_area_diff(mask1, mask2)
    p1 = crm.penalty_area_diff_account_parents(
        mask1, mask2, cell_container.color_to_cell, cell_container.parent_map
    )

    assert p1 < q1


def test_area_diff_with_mask():
    mask1, mask2, _ = produce_masks()

    m = crm.area_diff_mask(mask1, mask2)
    p1 = np.mean(m)
    p2 = crm.penalty_area_diff(mask1, mask2)

    assert np.abs(p1 - p2) < 1e-4


def test_overlap():
    m1, _, container1 = produce_masks(growth_rate=0.05)
    m2, _, container2 = produce_masks(growth_rate=0.03)
    iterations = container1.get_all_iterations()

    def get_overlap(container):
        i1 = iterations[1]
        cells = container.get_cells_at_iteration(i1)
        positions = [a[0].pos for a in cells.values()]
        positions = np.array(positions, dtype=np.float32)
        radii = np.array([a[0].radius for a in cells.values()], dtype=np.float32)
        return crm.overlap(positions, radii)

    o1 = get_overlap(container1)
    o2 = get_overlap(container2)

    assert o2 < 1e-3

    diff_mask = crm.parents_diff_mask(
        m1, m2, container2.color_to_cell, container2.parent_map, 0.5
    )
    parent_penalty = np.sum(diff_mask)

    assert parent_penalty > o1
    assert parent_penalty > o2
