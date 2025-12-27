from glob import glob
from pathlib import Path
import numpy as np

import cr_mech_coli as crm
from cr_mech_coli.crm_multilayer import MultilayerConfig


def run_sim(ml_config: MultilayerConfig) -> crm.CellContainer:
    positions = np.array(
        crm.generate_positions(
            n_agents=1,
            agent_settings=ml_config.agent_settings,
            config=ml_config.config,
            rng_seed=ml_config.rng_seed,
            dx=ml_config.dx,
            randomize_positions=ml_config.randomize_positions,
            n_vertices=ml_config.n_vertices,
        )
    )
    positions[:, :, 2] = 0.1 * ml_config.agent_settings.interaction.radius
    agent_dict = ml_config.agent_settings.to_rod_agent_dict()

    agents = [crm.RodAgent(p, 0.0 * p, **agent_dict) for p in positions]

    container = crm.run_simulation_with_agents(ml_config.config, agents)
    if container.path is not None:
        ml_config.to_toml_file(Path(container.path) / "ml_config.toml")
    else:
        print("Could not find save path for MultilayerConfig:")
        print(ml_config.to_toml_string())
    return container


def load_or_compute(
    ml_config: MultilayerConfig, out_path=Path("out/crm_multilayer/")
) -> crm.CellContainer:
    settings_files = glob(str(out_path / "*/ml_config.toml"))
    settings_files2 = glob(str(out_path / "*/*/ml_config.toml"))
    settings_files.extend(settings_files2)

    for file_path in settings_files:
        file_path = Path(file_path)
        ml_config_loaded = MultilayerConfig.load_from_toml_file(Path(file_path))
        if ml_config.approx_eq(ml_config_loaded):
            container = crm.CellContainer.load_from_storage(
                ml_config.config, file_path.parent
            )
            return container
    else:
        res = run_sim(ml_config)
        print()
        return res
