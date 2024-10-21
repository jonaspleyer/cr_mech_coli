import numpy as np

class MorsePotentialF32:
    """
    Interaction potential of our Agents.

    Famous :ref:`Morse <https://doi.org/10.1103/PhysRev.34.57>` potential for diatomic molecules.
    """
    radius: float
    potential_stiffness: float
    cutoff: float
    strength: float

class AgentSettings:
    """
    Contains settings needed to specify properties of the :class:`RodAgent`
    """
    mechanics: RodMechanicsSettings
    interaction: MorsePotentialF32
    growth_rate: float
    spring_length_threshold: float

    @staticmethod
    def __new__(cls, **kwargs) -> AgentSettings:
        ...

class CellIdentifier:
    """
    Unique identifier which is given to every cell in the simulation
    
    The identifier is comprised of the :class:`VoxelPlainIndex` in which the cell was first spawned.
    This can be due to initial setup or due to other methods such as division in a cell cycle.
    The second parameter is a counter which is unique for each voxel.
    This ensures that each cell obtains a unique identifier over the course of the simulation.
    """
    ...

class VoxelPlainIndex:
    """
    Identifier for voxels used internally to get rid of user-defined ones.
    """

class Configuration:
    """
    Contains all settings needed to configure the simulation
    """
    agent_settings: AgentSettings
    n_agents: int
    n_threads: int
    t0: float
    dt: float
    t_max: float
    save_interval: float
    show_progressbar: bool
    domain_size: float
    domain_height: float
    randomize_position: float
    n_voxels: int
    rng_seed: int

    @staticmethod
    def __new__(cls, **kwargs) -> Configuration: ...
    @staticmethod
    def from_json(json_string: str) -> Configuration: ...
    def to_json(self) -> str: ...
    def to_hash(self) -> int: ...

class RodAgent:
    """
    A basic cell-agent which makes use of
    `RodMechanics <https://cellular-raza.com/docs/cellular_raza_building_blocks/structs.RodMechanics.html>`_
    """
    pos: np.ndarray
    vel: np.ndarray
    radius: float
    growth_rate: float
    spring_length_threshold: float
    def __repr__(self) -> str: ...

class RodMechanicsSettings:
    """
    Contains all settings required to construct :class:`RodMechanics`
    """
    pos: np.ndarray
    vel: np.ndarray
    diffusion_constant: float
    spring_tension: float
    angle_stiffness: float
    spring_length: float
    damping: float

class SimResult:
    """
    Resulting type when executing a full simulation
    """
    def get_cells(self) -> dict[int, dict[CellIdentifier, RodAgent]]: ...
    def get_cells_at_iteration(self, iteration: int) -> tuple[
        dict[CellIdentifier, RodAgent], CellIdentifier | None
    ]: ...
    def get_cell_history(self, identifier: CellIdentifier) -> dict[int, RodAgent]: ...

def run_simulation(config: Configuration) -> SimResult:
    """
    Executes the simulation with the given :class:`Configuration`
    """
    ...

def sort_cellular_identifiers(identifiers: list[CellIdentifier]) -> list[CellIdentifier]:
    """
    Sorts an iterator of :class:`CellIdentifier` deterministically.
    
    This function is usefull for generating identical masks every simulation run.
    This function is implemented as standalone since sorting of a :class:`CellIdentifier` is
    typically not supported.
    
    Args:
        identifiers(list): A list of :class:`CellIdentifier`
    
    Returns:
        list: The sorted list.
    """
    ...

