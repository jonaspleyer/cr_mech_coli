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


def test_render_mask_2d():
    _, _, container = produce_masks()
    config = __create_config()
    i1 = container.get_all_iterations()[1]

    resolution = (int(12.8 * config.domain_size[0]), int(12.8 * config.domain_size[1]))

    import time

    times = []
    for _ in range(5):
        t = time.process_time()
        mask1, overlap_mask = crm.render_mask_2d(
            container.get_cells_at_iteration(i1),
            container.cell_to_color,
            (config.domain_size[0], config.domain_size[1]),
            resolution,
            delta_angle=np.float32(np.pi / 15.0),
        )
        t1 = time.process_time() - t
        t = time.process_time()

        mask2 = crm.render_mask(
            container.get_cells_at_iteration(i1),
            container.cell_to_color,
            (config.domain_size[0], config.domain_size[1]),
        )
        t2 = time.process_time() - t

        times.append([t1, t2])

        _ = crm.parents_diff_mask(
            mask1,
            mask2,
            container.color_to_cell,
            container.parent_map,
            0.5,
        )

        # Remove overlapping regions from second mask
        mask2[np.any(overlap_mask != 0, axis=2)] = [0, 0, 0]
        # Compare both masks pixel-wise
        assert np.sum(np.any(mask2 != mask1, axis=2)) < 2000

        n1 = np.unique(mask1.reshape((-1, 3)), axis=0)
        n2 = np.unique(mask2.reshape((-1, 3)), axis=0)
        assert np.all(n1 == n2)

    means = np.mean(times, axis=0)
    std = np.std(times, axis=0)
    # Assert that method is at least 10 times faster with 1sigma probability
    t_pyvista = means[1] - std[1]
    t_plotters = means[0] + std[0]
    assert t_plotters * 10 < t_pyvista


def test_render_mask_2d_2():
    # This test originated from a bug that was found within the calculate_polygon_hull method
    # where too many spherical coordinates were inserted in the middle of a rod
    config = crm.Configuration()
    agent_settings = crm.AgentSettings(
        growth_rate=0.078947365,
    )
    config.t0 = 0.0
    config.dt = 0.1
    config.t_max = 200.0
    config.n_saves = 10 - 2
    config.domain_size = (np.float32(150.0), np.float32(150.0))

    positions = crm.generate_positions(
        n_agents=4,
        agent_settings=agent_settings,
        config=config,
        rng_seed=1,
        dx=(config.domain_size[0] * 0.1, config.domain_size[1] * 0.1),
    )
    agent_dict = agent_settings.to_rod_agent_dict()
    cells = [crm.RodAgent(p, 0.0 * p, **agent_dict) for p in positions]

    container = crm.run_simulation_with_agents(config, cells)

    i1 = container.get_all_iterations()[2]
    k, v = list(container.get_cells_at_iteration(i1).items())[1]
    cells = {k: v}

    m1, m2 = crm.render_mask_2d(
        cells,
        container.cell_to_color,
        (config.domain_size[0], config.domain_size[1]),
        (int(config.domain_size[0] * 10), int(config.domain_size[1] * 10)),
        delta_angle=np.float32(np.pi / 15.0),
    )

    # Perform very crude thinning of non-zero values
    filt = np.any(m1 != 0, axis=2)
    filtnew = np.copy(filt)
    filtnew[2:] *= filt[:-2]
    filtnew[:-2] *= filt[2:]
    filtnew[:, 2:] *= filt[:, :-2]
    filtnew[:, :-2] *= filt[:, 2:]

    import cv2 as cv

    n_components, _ = cv.connectedComponents(filtnew.astype(np.uint8))
    assert n_components >= 3
