import cr_mech_coli as crm


def test_config_set_attributes_3():
    config = crm.Configuration(domain_size=[1000, 900])
    assert abs(config.domain_size[0] - 1000) < 1e-8
    assert abs(config.domain_size[1] - 900) < 1e-8


def test_config_set_attributes_4():
    config = crm.Configuration(n_threads=2)
    assert config.n_threads == 2


def test_config_set_attributes_5():
    config = crm.Configuration(domain_height=10.0)
    assert abs(config.domain_height - 10.0) < 1e-8


def test_config_set_attributes_6():
    config = crm.Configuration(n_voxels=[3, 2])
    assert config.n_voxels == [3, 2]


def test_agent_settings():
    agent_settings = crm.AgentSettings(
        growth_rate_setter={"mean": 0.003, "std": 937.85}
    )
    assert abs(agent_settings.growth_rate_setter.mean - 0.003) < 1e-3
    assert abs(agent_settings.growth_rate_setter.std - 937.85) < 1e-3

    agent_settings = crm.AgentSettings(growth_rate_setter={"g1": 0.03, "g2": -0.1})
    assert abs(agent_settings.growth_rate_setter.g1 - 0.03) < 1e-5
    assert abs(agent_settings.growth_rate_setter.g2 + 0.1) < 1e-5


def test_interaction_assign():
    agent_settings = crm.AgentSettings()
    print(agent_settings.interaction)
    agent_settings.interaction.strength = 0.1091
    assert abs(agent_settings.interaction.strength - 0.1091) < 1e-4
    agent_settings.interaction.potential_stiffness = 11.1
    assert abs(agent_settings.interaction.potential_stiffness - 11.1) < 1e-3
    agent_settings.interaction.cutoff = 33.2391
    assert abs(agent_settings.interaction.cutoff - 33.2391) < 1e-5
    agent_settings.interaction.radius = 1.93e-8
    assert abs(agent_settings.interaction.radius - 1.93e-8) < 1e-10
