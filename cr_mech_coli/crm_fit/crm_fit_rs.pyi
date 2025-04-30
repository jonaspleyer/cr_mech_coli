import enum

import cr_mech_coli as crm

class PotentialType(enum.Enum):
    Mie = 0
    Morse = 1

class PotentialType_Mie:
    """
    Variant of the :class:`PotentialType`.
    """

    ...

class PotentialType_Morse: ...

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
    potential_type: PotentialType

class Constants:
    t_max: float
    dt: float
    domain_size: float
    n_voxels: int
    rng_seed: int
    cutoff: float
    pixel_per_micron: float
    n_vertices: int

class Optimization:
    seed: int
    tol: float
    max_iter: int
    pop_size: int
    recombination: float

class Settings:
    parameters: Parameters
    constants: Constants
    optimization: Optimization

    @staticmethod
    def from_toml(filename: str) -> Settings: ...
    @staticmethod
    def from_toml_string(toml_str: str) -> Settings: ...
    def to_config(self, n_saves: int) -> crm.Configuration: ...
    def generate_optimization_infos(self, n_agents: int) -> list: ...
