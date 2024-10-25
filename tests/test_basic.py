import cr_mech_coli as crm
import numpy as np
from concurrent.futures import ProcessPoolExecutor

def run_config(config):
    config = crm.Configuration.from_json(config)
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
    )

if __name__ == "__main__":
    configs = [
        crm.Configuration(growth_rate=rate, n_agents=n_agents).to_json()
        for rate in np.linspace(0.0, 0.1, 20)
        for n_agents in np.arange(3, 6)
    ]

    pool = ProcessPoolExecutor(max_workers=14)

    _ = pool.map(run_config, configs)
