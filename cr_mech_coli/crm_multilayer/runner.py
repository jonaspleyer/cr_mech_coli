import cr_mech_coli as crm
from pathlib import Path
import numpy as np
from glob import glob

from cr_mech_coli.crm_multilayer import MultilayerConfig
from cr_mech_coli import GrowthRateSetter


def produce_ml_config() -> MultilayerConfig:
    """
    Produces a :class:`MultilayerConfig` with default parameters.
    """
    ml_config = crm.crm_multilayer.MultilayerConfig()

    # TIME SETTINGS
    ml_config.config.dt = 0.05
    ml_config.config.t_max = 1800
    ml_config.config.dt = 0.025
    ml_config.config.n_saves = 19

    # SOLVER SETTINGS
    ml_config.config.n_threads = 1
    ml_config.config.rng_seed = 0
    ml_config.config.progressbar = None
    ml_config.config.storage_options = [
        crm.simulation.StorageOption.Memory,
        crm.simulation.StorageOption.SerdeJson,
    ]
    ml_config.config.storage_location = "out/crm_multilayer"

    # DOMAIN SETTINGS
    ml_config.config.domain_height = 20.0
    ml_config.config.domain_size = (400, 400)
    ml_config.dx = (100, 100)
    # ml_config.config.domain_size = (1600, 1600)
    # ml_config.dx = (700, 700)
    ml_config.config.n_voxels = (10, 10)

    # EXTERNAL FORCES
    ml_config.config.gel_pressure = 0.05
    ml_config.config.surface_friction = 0.3
    ml_config.config.surface_friction_distance = (
        ml_config.agent_settings.interaction.radius / 10
    )

    # AGENT SETTINGS
    ## GROWTH
    ml_config.agent_settings.neighbor_reduction = (200, 0.5)
    ml_config.agent_settings.growth_rate = 0.005
    ml_config.agent_settings.growth_rate_setter = GrowthRateSetter.NormalDistr(
        0.005, 0.001
    )

    # MECHANICS
    ml_config.agent_settings.mechanics.damping = 0.02
    ml_config.agent_settings.mechanics.diffusion_constant = 0.03
    ml_config.agent_settings.mechanics.rigidity = 1.0
    ml_config.agent_settings.mechanics.spring_tension = 0.3

    # INTERACTION
    ml_config.agent_settings.interaction.strength = 0.02
    ml_config.agent_settings.spring_length_threshold = 20.0

    return ml_config


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
