import numpy as np

import cr_mech_coli as crm
import cr_mech_coli.crm_multilayer as crmm


def test_produce_ml_config(ret=False):
    return crmm.produce_ml_config() if ret else None


def test_run_default_ml_config(ml_config=None, ret=False):
    if ml_config is None:
        ml_config = test_produce_ml_config(ret=True)
    ml_config.config.t_max = 1.0
    ml_config.config.storage_options = [crm.simulation.StorageOption.Memory]
    container = crmm.run_sim(ml_config, False)
    return container if ret else None


def test_produce_ydata():
    ml_config = test_produce_ml_config(ret=True)
    container = test_run_default_ml_config(ml_config, ret=True)
    iterations, positions, ymax, y95th, ymean = crmm.produce_ydata(container)

    assert len(iterations) == ml_config.config.n_saves + 2
    assert len(iterations) == len(positions)
    for i, p in zip(iterations, positions):
        n_cells = len(container.get_cells_at_iteration(i))
        assert p.shape == (n_cells, ml_config.n_vertices, 3)

    assert ymax.shape == (len(iterations),)
    assert y95th.shape == (len(iterations),)
    assert ymean.shape == (len(iterations),)


def set_strength(ml_config, x):
    ml_config.agent_settings.interaction.strength = x


def __generate_arguments():
    return [
        ("agent_settings.interaction.strength", 0.0, 10.0, 3),
        ("agent_settings.mechanics.damping", 1.0, 5.0, 5),
        ("agent_settings.mechanics.rigidity", 1.0, 10.0, 4, "log"),
        ("config.gel_pressure", 0.0, 0.1, 7),
    ]


def test_sample_parameters():
    ml_configs = crmm.sample_parameters(*__generate_arguments())
    ml_configs = list(ml_configs)

    # Check length of generated configs
    assert len(ml_configs) == 3 * 5 * 4 * 7

    for ml_config in ml_configs:
        x = ml_config.agent_settings.interaction.strength
        assert x == 0.0 or x == 5.0 or x == 10.0
        y = ml_config.agent_settings.mechanics.damping
        assert y == 1.0 or y == 2.0 or y == 3.0 or y == 4.0 or y == 5.0
        z = ml_config.agent_settings.mechanics.rigidity
        x0 = np.log(1)
        x1 = np.log(10)
        dx = (x1 - x0) / 3
        assert (
            z == 1.0
            or np.abs(z - np.exp(x0 + dx)) < 1e-4
            or np.abs(z - np.exp(x0 + 2 * dx)) < 1e-4
            or np.abs(z - 10.0) < 1e-4
        )
        h = ml_config.config.gel_pressure
        assert np.abs(h * 6 - np.round(h * 6, 1)) < 1e-6


def test_produce_ydata_samples1():
    results1 = crmm.load_or_compute_ydata_samples([], n_threads_total=1)
    assert results1 == []


def test_produce_ydata_samples2():
    ml_config = crmm.produce_ml_config(
        ("config.t_max", 150.0),
        ("config.storage_options", [crm.StorageOption.Memory]),
    )
    ml_config.config.progressbar = None
    ml_configs = list(
        crmm.sample_parameters(*__generate_arguments(), ml_config_default=ml_config)
    )

    results2 = crmm.load_or_compute_ydata_samples(ml_configs[::10], n_threads_total=1)
    assert len(results2) == len(ml_configs[::10])
