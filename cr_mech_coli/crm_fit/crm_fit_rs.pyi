import enum
import numpy as np
from pathlib import Path

import cr_mech_coli as crm
from cr_mech_coli.cr_mech_coli import CellContainer

class PotentialType(enum.Enum):
    Mie = 0
    Morse = 1

class SampledFloat:
    min: float
    max: float
    initial: float
    individual: bool | None

    @staticmethod
    def __new__(
        cls, min: float, max: float, initial: float, individual: bool | None = None
    ) -> SampledFloat: ...

class Parameter(enum.Enum):
    SampledFloat = dict
    Float = float
    List = list

class Parameters:
    radius: Parameter | SampledFloat | list | float
    rigidity: Parameter | SampledFloat | list | float
    damping: Parameter | SampledFloat | list | float
    strength: Parameter | SampledFloat | list | float
    growth_rate: Parameter | SampledFloat | list | float
    potential_type: PotentialType

class Constants:
    t_max: float
    dt: float
    domain_size: tuple[float, float]
    n_voxels: int
    rng_seed: int
    cutoff: float
    pixel_per_micron: float
    n_vertices: int
    n_saves: int

class DifferentialEvolution:
    seed: int
    tol: float
    max_iter: int
    pop_size: int
    recombination: float

class OptimizationMethod(enum.Enum):
    DifferentialEvolution = DifferentialEvolution()

class Others:
    show_progressbar: bool
    @staticmethod
    def __new__(cls, show_progressbar: bool = False) -> SampledFloat: ...

class OptimizationInfos:
    bounds_lower: list[float]
    bounds_upper: list[float]
    initial_values: list[float]
    parameter_infos: list[tuple[str, str, str]]
    constants: list[float]
    constant_infos: list[tuple[str, str, str]]

class Settings:
    parameters: Parameters
    constants: Constants
    optimization: OptimizationMethod
    others: Others

    @staticmethod
    def from_toml(filename: str) -> Settings: ...
    @staticmethod
    def from_toml_string(toml_str: str) -> Settings: ...
    def to_config(self, n_saves: int) -> crm.Configuration: ...
    def generate_optimization_infos(self, n_agents: int) -> OptimizationInfos: ...
    def get_final_param(
        self,
        param_name: str,
        optimization_result: OptimizationResult,
        n_agents: int,
        agent_index: int,
    ) -> float: ...

def run_simulation(
    parameters: list[float], positions: np.ndarray, settings: Settings
) -> CellContainer: ...

class OptimizationResult:
    params: list[float]
    cost: float
    success: bool | None
    neval: int | None
    niter: int | None

    def save_to_file(self, filename: Path): ...
    @staticmethod
    def load_from_file(filename: Path): ...

def run_optimizer(
    iterations: np.ndarray,
    positions: np.ndarray,
    settings: Settings,
    n_workers: int = -1,
) -> OptimizationResult: ...
