import cr_mech_coli as crm
from glob import glob


def test_store_iamges():
    config = crm.Configuration()
    config.t0 = 0.0
    config.dt = 0.1
    config.t_max = 40.0
    config.n_saves = 1

    agent_settings = crm.AgentSettings()
    agent_settings.growth_rate = 0.1

    positions = crm.generate_positions_old(8, agent_settings, config, rng_seed=1031)
    agents = [
        crm.RodAgent(pos=p, vel=0 * p, **agent_settings.to_rod_agent_dict())
        for p in positions
    ]

    cell_container = crm.run_simulation_with_agents(config, agents)
    render_settings = crm.RenderSettings()
    render_settings.noise = 50
    render_settings.kernel_size = 30
    render_settings.ssao_radius = 50

    save_dir = "./tests/test_image_gen_basic/"
    crm.store_all_images(
        cell_container,
        config.domain_size,
        render_settings,
        render_raw_pv=True,
        save_dir=save_dir,
    )

    # Check that the files exist
    assert len(glob(f"{save_dir}images/*.png")) == 2
    assert len(glob(f"{save_dir}masks/*.png")) == 2
    assert len(glob(f"{save_dir}raw_pv/*.png")) == 2
