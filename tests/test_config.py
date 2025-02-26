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
