import cr_mech_coli as crm

if __name__ == "__main__":
    config = crm.Configuration(
        n_agents=4,
        growth_rate=0.078947365,
    )
    config.t0 = 0.0
    config.dt = 0.1
    config.t_max = 100.0
    config.save_interval = 20.0

    sim_result = crm.run_simulation(config)
    render_settings = crm.RenderSettings()
    render_settings.noise = 50
    render_settings.kernel_size = 30
    render_settings.ssao_radius = 50

    crm.store_all_images(
        config,
        sim_result,
        render_settings,
        render_raw_pv=True,
        use_hash=True,
        save_dir="docs/source/_static/",
    )