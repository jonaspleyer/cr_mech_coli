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
