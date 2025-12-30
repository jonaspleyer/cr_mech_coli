from typing import Any
import cr_mech_coli as crm
from pathlib import Path
import numpy as np
from glob import glob
import multiprocessing as mp
import itertools
from tqdm import tqdm

from cr_mech_coli.crm_multilayer import MultilayerConfig
from cr_mech_coli import GrowthRateSetter


def produce_ml_config(*args: tuple[str, Any]) -> MultilayerConfig:
    """
    Produces a :class:`MultilayerConfig` with default parameters.
    """
    ml_config = crm.crm_multilayer.MultilayerConfig()

    # TIME SETTINGS
    ml_config.config.dt = 0.05
    ml_config.config.t_max = 3000
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
    # ml_config.config.domain_size = (400, 400)
    ml_config.dx = (100, 100)
    ml_config.config.domain_size = (1600, 1600)
    ml_config.dx = (200, 200)
    ml_config.config.n_voxels = (40, 40)

    # EXTERNAL FORCES
    ml_config.config.gel_pressure = 0.10
    ml_config.config.surface_friction = 0.03
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
    ml_config.agent_settings.mechanics.rigidity = 20.0
    ml_config.agent_settings.mechanics.spring_tension = 10.0

    # INTERACTION
    ml_config.agent_settings.interaction.potential_stiffness = 1.0

    for attr, value in args:
        attrs = attr.split(".")
        base = ml_config
        for a in attrs[:-1]:
            base = base.__getattribute__(a)
        base.__setattr__(attrs[-1], value)

    # INTERACTION
    ml_config.agent_settings.interaction.strength = 0.03
    ml_config.agent_settings.spring_length_threshold = 20.0

    return ml_config


def run_sim(ml_config: MultilayerConfig, store_positions=True) -> crm.CellContainer:
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

    if store_positions and container.path is not None:
        opath = container.path
        opath.mkdir(parents=True, exist_ok=True)

        # Calculate data and store it to files
        iterations, positions, _, _, _ = produce_ydata(container)
        opath = opath / "calculated"
        opath.mkdir(parents=True, exist_ok=True)
        np.save(opath / "iterations.npy", iterations)
        for i, p in enumerate(positions):
            np.save(opath / f"positions-{i:05}.npy", p)

    return container


def find_ml_config_path(
    ml_config: MultilayerConfig, out_path=Path("out/crm_multilayer/")
):
    settings_files = glob(str(out_path / "*/ml_config.toml"))
    settings_files2 = glob(str(out_path / "*/*/ml_config.toml"))
    settings_files.extend(settings_files2)

    for file_path in settings_files:
        file_path = Path(file_path)
        ml_config_loaded = MultilayerConfig.load_from_toml_file(Path(file_path))
        if ml_config.approx_eq(ml_config_loaded):
            return file_path
    return None


def load_or_compute_container(
    ml_config: MultilayerConfig,
    out_path=Path("out/crm_multilayer/"),
    store_positions=True,
) -> crm.CellContainer:
    settings_files = glob(str(out_path / "*/ml_config.toml"))
    settings_files2 = glob(str(out_path / "*/*/ml_config.toml"))
    settings_files.extend(settings_files2)

    file_path = find_ml_config_path(ml_config, out_path)
    if file_path is not None:
        container = crm.CellContainer.load_from_storage(
            ml_config.config, file_path.parent
        )
        return container
    else:
        res = run_sim(ml_config, store_positions=store_positions)
        if res.path is not None:
            print()
        return res


def __calculate_properties(positions: list[np.ndarray]):
    ymax = np.array([np.max(p[:, :, 2]) for p in positions])
    y95th = np.array([np.percentile(p[:, :, 2], 95) for p in positions])
    ymean = np.array([np.mean(p[:, :, 2]) for p in positions])
    return ymax, y95th, ymean


def produce_ydata(container: crm.CellContainer):
    cells = container.get_cells()
    iterations = np.array(container.get_all_iterations())
    positions = [np.array([c[0].pos for c in cells[i].values()]) for i in iterations]
    ymax, y95th, ymean = __calculate_properties(positions)
    return iterations, positions, ymax, y95th, ymean


def load_or_compute_ydata(
    ml_config: MultilayerConfig,
    out_path=Path("out/crm_multilayer"),
    store_positions=True,
):
    positions = []
    file_path = find_ml_config_path(ml_config, out_path)
    if file_path is not None:
        # First try to load data directly
        try:
            positions = []
            opath = file_path / "calculated"
            iterations = np.load(str(opath / "iterations.npy"))
            for file in sorted(glob(str(opath / "positions-*.npy"))):
                p = np.load(file)
                positions.append(p)
            ymax, y95th, ymean = __calculate_properties(positions)
        except:
            # Then try to load container
            container = crm.CellContainer.load_from_storage(
                ml_config.config, file_path.parent
            )
            iterations, positions, ymax, y95th, ymean = produce_ydata(container)

    # Otherwise calculate new result
    else:
        container = load_or_compute_container(
            ml_config, out_path, store_positions=store_positions
        )
        iterations, positions, ymax, y95th, ymean = produce_ydata(container)

    return iterations, positions, ymax, y95th, ymean


def __create_full_sample(value) -> tuple[float, float, int, str]:
    if len(value) == 3:
        low, high, n = value
        sample_type = "Uniform"
    elif len(value) == 4:
        low, high, n, sample_type = value
    else:
        raise TypeError(
            "Expected tuple with (low, high, n) or (low, high, n, sample_type)"
        )
    return low, high, n, sample_type


def __create_sample(
    low: float, high: float, n: int, sample_type: str | None
) -> np.ndarray:
    if sample_type is None or sample_type.lower() == "uniform":
        return np.linspace(low, high, n, dtype=np.float32)
    elif sample_type.lower() == "log":
        x0 = np.log(low)
        x1 = np.log(high)
        return np.exp(np.linspace(x0, x1, n, dtype=np.float32))
    else:
        raise KeyError("could not identify sample type")


def __set_ml_config(ml_config, setter, p):
    sets = setter.split(".")
    base = ml_config
    for var in sets[:-1]:
        base = base.__getattribute__(var)
    base.__setattr__(sets[-1], p)


def sample_parameters(
    *args: tuple[Any, float, float, float] | tuple[Any, float, float, int, str],
    ml_config_default: MultilayerConfig | None = None,
):
    param_setters = []
    samples_all = []

    if len(args) == 0:
        return None

    for value in args:
        sample_info = __create_full_sample([value[i] for i in range(1, len(value))])
        sample = __create_sample(*sample_info)
        param_setters.append(value[0])
        samples_all.append(sample)

    # Build combined sample
    # TODO For the future: do this without allocating the entire array
    samples = np.empty(
        [len(x) for x in samples_all] + [len(samples_all)], dtype=np.float32
    )
    for i, a in enumerate(np.ix_(*samples_all)):
        samples[..., i] = a

    for param_combination in samples.reshape(-1, len(samples_all)):
        pass
        # Assign parameters to new multilayerconfig here
        if ml_config_default is not None:
            ml_config = ml_config_default.clone_with_args()
        else:
            ml_config = produce_ml_config()
        for setter, p in zip(param_setters, param_combination):
            __set_ml_config(ml_config, setter, p)
        yield ml_config


def __run_helper(args):
    (ml_config_str, out_path, store_positions) = args
    ml_config = MultilayerConfig.load_from_toml_str(ml_config_str)
    load_or_compute_ydata(ml_config, out_path, store_positions)


def load_or_compute_ydata_samples(
    ml_configs,
    n_threads_total: int | None = None,
    out_path=Path("out/crm_multilayer"),
    store_positions=True,
    show_progressbar=True,
):
    if n_threads_total is None:
        n_threads_total = mp.cpu_count()

    # Calculate multiple results with a pool
    pool = mp.Pool(n_threads_total)
    n_samples = len(ml_configs)
    arglist = zip(
        (m.to_toml_string() for m in ml_configs),
        itertools.repeat(out_path),
        itertools.repeat(store_positions),
    )
    results = list(
        tqdm(
            pool.imap(__run_helper, arglist),
            total=n_samples,
            disable=not show_progressbar,
        )
    )

    return results
