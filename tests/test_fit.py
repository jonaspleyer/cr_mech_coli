import cr_mech_coli as crm
from cr_mech_coli.imaging import RenderSettings

def produce_masks():
    config = crm.Configuration()
    config.agent_settings.growth_rate = 0.05
    config.t0 = 0.0
    config.dt = 0.1
    config.t_max = 100.0
    config.save_interval = 20.0
    config.n_agents = 4
    config.show_progressbar = True

    sim_result = crm.run_simulation(config)

    all_cells = sim_result.get_cells()
    iterations = sim_result.get_all_iterations()
    colors = sim_result.cell_to_color
    i1 = iterations[0]
    i2 = iterations[1]

    rs = RenderSettings(resolution=200)
    mask1 = crm.render_mask(config, all_cells[i1], colors, render_settings=rs)
    mask2 = crm.render_mask(config, all_cells[i2], colors, render_settings=rs)
    return mask1, mask2, sim_result


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
    mask1, mask2, sim_result = produce_masks()

    p1 = crm.penalty_area_diff_account_parents(mask1, mask2, sim_result)
    p2 = crm.penalty_area_diff_account_parents(mask1, mask1, sim_result)
    p3 = crm.penalty_area_diff_account_parents(mask2, mask2, sim_result)

    assert p1 > 0
    assert p2 == 0
    assert p3 == 0

    assert p1 <= 1
    assert p2 <= 1
    assert p3 <= 1

def test_area_diff_comparison():
    mask1, mask2, parent_map = produce_masks()

    p1 = crm.penalty_area_diff_account_parents(mask1, mask2, parent_map)
    q1 = crm.penalty_area_diff(mask1, mask2)

    assert p1 < q1
