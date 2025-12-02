import enum
import numpy as np
from pathlib import Path

import cr_mech_coli as crm
from cr_mech_coli.cr_mech_coli import CellContainer

class PotentialType(enum.Enum):
    Mie = 0
    Morse = 1

class SampledFloat:
    min: np.float32
    max: np.float32
    initial: np.float32
    individual: bool | None

    @staticmethod
    def __new__(
        cls,
        min: np.float32,
        max: np.float32,
        initial: np.float32,
        individual: bool | None = None,
    ) -> SampledFloat: ...

class Parameter(enum.Enum):
    SampledFloat = dict
    Float = np.float32
    List = list

class Parameters:
    radius: Parameter | SampledFloat | list | np.float32
    rigidity: Parameter | SampledFloat | list | np.float32
    damping: Parameter | SampledFloat | list | np.float32
    strength: Parameter | SampledFloat | list | np.float32
    growth_rate: Parameter | SampledFloat | list | np.float32
    potential_type: PotentialType

class Constants:
    t_max: np.float32
    dt: np.float32
    domain_size: tuple[np.float32, np.float32]
    n_voxels: int
    rng_seed: int
    cutoff: np.float32
    pixel_per_micron: np.float32
    n_vertices: int
    n_saves: int
    displacement_error: np.float32

class DifferentialEvolution:
    seed: int
    tol: np.float32
    max_iter: int
    pop_size: int
    recombination: np.float32

class OptimizationMethod(enum.Enum):
    DifferentialEvolution = DifferentialEvolution()

class Others:
    progressbar: str | None
    @staticmethod
    def __new__(cls, progressbar: str | None = None) -> SampledFloat: ...

class OptimizationInfos:
    bounds_lower: list[np.float32]
    bounds_upper: list[np.float32]
    initial_values: list[np.float32]
    parameter_infos: list[tuple[str, str, str]]
    constants: list[np.float32]
    constant_infos: list[tuple[str, str, str]]

class Settings:
    parameters: Parameters
    constants: Constants
    optimization: OptimizationMethod
    others: Others
    domain_height: np.float32

    @staticmethod
    def from_toml(filename: str | Path) -> Settings: ...
    @staticmethod
    def from_toml_string(toml_str: str) -> Settings: ...
    def to_config(self) -> crm.Configuration: ...
    def generate_optimization_infos(self, n_agents: int) -> OptimizationInfos: ...
    def get_final_param(
        self,
        param_name: str,
        optimization_result: OptimizationResult,
        n_agents: int,
        agent_index: int,
    ) -> np.float32: ...
    def get_param(
        self,
        param_name: str,
        optimization_result: OptimizationResult,
        n_agents: int,
        agent_index: int,
    ) -> np.float32: ...

def run_simulation(
    parameters: list[np.float32], positions: np.ndarray, settings: Settings
) -> CellContainer: ...

class OptimizationResult:
    params: list[np.float32]
    cost: np.float32
    success: bool | None
    neval: int | None
    niter: int | None
    evals: list[int]

    def save_to_file(self, filename: Path): ...
    @staticmethod
    def load_from_file(filename: Path): ...

def run_optimizer(
    iterations: np.ndarray,
    positions: np.ndarray,
    settings: Settings,
    n_workers: int = -1,
) -> OptimizationResult: ...
