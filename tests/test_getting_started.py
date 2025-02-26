import cr_mech_coli as crm


def test_getting_started():
    config = crm.Configuration()
    agent_settings = crm.AgentSettings()
    positions = crm.generate_positions_old(4, agent_settings, config)
    agents = [
        crm.RodAgent(pos=p, vel=p * 0.0, **agent_settings.to_rod_agent_dict())
        for p in positions
    ]

    cell_container = crm.run_simulation_with_agents(config, agents)

    crm.store_all_images(cell_container, config.domain_size)
