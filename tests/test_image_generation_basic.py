import cr_mech_coli as crm
from glob import glob


def test_store_iamges():
    config = crm.Configuration()
    config.t0 = 0.0
    config.dt = 0.1
    config.t_max = 40.0
    config.save_interval = 20.0
    config.n_agents = 5

    agent_settings = crm.AgentSettings()
    agent_settings.growth_rate = 0.02

    cell_container = crm.run_simulation(config, agent_settings)
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
