import cr_mech_coli as crm


def test_getting_started():
    config = crm.Configuration()
    agent_settings = crm.AgentSettings()

    agents = crm.generate_agents(4, agent_settings, config)

    cell_container = crm.run_simulation_with_agents(config, agents)

    crm.store_all_images(cell_container, config.domain_size)
