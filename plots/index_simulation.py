import cr_mech_coli as crm

if __name__ == "__main__":
    config = crm.Configuration()
    agent_settings = crm.AgentSettings(
        growth_rate=0.078947365,
    )
    config.t0 = 0.0
    config.dt = 0.1
    config.t_max = 100.0
    config.n_saves = 4
    config.domain_size = (150, 150)

    positions = crm.generate_positions(
        n_agents=4,
        agent_settings=agent_settings,
        config=config,
        rng_seed=3,
        dx=(config.domain_size[0] * 0.1, config.domain_size[1] * 0.1),
    )
    agent_dict = agent_settings.to_rod_agent_dict()
    agents = [crm.RodAgent(p, 0.0 * p, **agent_dict) for p in positions]

    sim_result = crm.run_simulation_with_agents(config, agents)
    render_settings = crm.RenderSettings()
    render_settings.noise = 50
    render_settings.kernel_size = 30
    render_settings.ssao_radius = 50

    crm.store_all_images(
        sim_result,
        config.domain_size,
        render_settings,
        render_raw_pv=True,
        save_dir="docs/source/_static/",
        store_config=config,
        use_hash=True,
    )
