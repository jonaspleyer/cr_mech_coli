import numpy as np
from pathlib import Path
from enum import Enum

class StorageOption(Enum):
    Sled = 0
    SledTemp = 1
    SerdeJson = 2
    Ron = 3
    Memory = 4

class PhysicalInteraction: ...

class MiePotentialF32:
    """\
    See
    :url:`https://cellular-raza.com/docs/cellular_raza_building_blocks/struct.MiePotentialF32.html`.
    """

    radius: np.float32
    potential_stiffness: np.float32
    cutoff: np.float32
    strength: np.float32

    @staticmethod
    def __new__(
        cls, radius: float, strength: float, bound: float, cutoff: float, en: float, em
    ) -> MorsePotentialF32: ...

class MorsePotentialF32:
    """\
    Interaction potential of our Agents.

    Famous :ref:`Morse <https://doi.org/10.1103/PhysRev.34.57>` potential for diatomic molecules.
    """

    radius: np.float32
    potential_stiffness: np.float32
    cutoff: np.float32
    strength: np.float32

    @staticmethod
    def __new__(cls, **kwargs) -> MorsePotentialF32: ...

class SpringLengthThresholdSetter:
    l1: np.float32
    l2: np.float32
    mean: np.float32
    std: np.float32

class GrowthRateSetter:
    g1: np.float32
    g2: np.float32
    mean: np.float32
    std: np.float32

class AgentSettings:
    """\
    Contains settings needed to specify properties of the :class:`RodAgent`
    """

    mechanics: RodMechanicsSettings
    interaction: MorsePotentialF32
    growth_rate: np.float32
    growth_rate_setter: GrowthRateSetter
    spring_length_threshold: np.float32
    neighbor_reduction: tuple[int, np.float32] | None

    @staticmethod
    def __new__(cls, **kwargs) -> AgentSettings: ...
    def to_rod_agent_dict(self) -> dict: ...

class CellIdentifier:
    """
    Unique identifier which is given to every cell in the simulation

    The identifier is comprised of the :class:`VoxelPlainIndex` in which the cell was first spawned.
    This can be due to initial setup or due to other methods such as division in a cell cycle.
    The second parameter is a counter which is unique for each voxel.
    This ensures that each cell obtains a unique identifier over the course of the simulation.
    """

    ...
    @staticmethod
    def __new__(cls, voxel_index: VoxelPlainIndex, countere: int) -> CellIdentifier: ...
    @staticmethod
    def new_initial(counter: int) -> CellIdentifier: ...
    def __getitem__(self, key) -> int: ...

class VoxelPlainIndex:
    """\
    Identifier for voxels used internally to get rid of user-defined ones.
    """
    @staticmethod
    def __new__(cls, index) -> VoxelPlainIndex: ...

class Configuration:
    """\
    Contains all settings needed to configure the simulation
    """

    agent_settings: AgentSettings
    n_agents: int
    n_threads: int
    t0: np.float32 | float
    dt: np.float32 | float
    t_max: np.float32 | float
    n_saves: int
    progressbar: str | None
    domain_size: tuple[np.float32, np.float32] | tuple[float, float]
    domain_height: np.float32 | float
    n_voxels: tuple[int, int]
    rng_seed: int
    gel_pressure: np.float32 | float
    surface_friction: np.float32 | float
    surface_friction_distance: np.float32 | float
    storage_options: list[StorageOption]
    storage_location: Path | str
    storage_suffix: Path | str | None = None

    @staticmethod
    def __new__(cls, **kwargs) -> Configuration: ...
    @staticmethod
    def from_json(json_string: str) -> Configuration: ...
    def to_json(self) -> str: ...
    def to_hash(self) -> int: ...
    @staticmethod
    def from_toml(toml_string) -> Configuration: ...

class RodAgent:
    """\
    A basic cell-agent which makes use of
    `RodMechanics <https://cellular-raza.com/docs/cellular_raza_building_blocks/structs.RodMechanics.html>`_
    """

    pos: np.ndarray
    vel: np.ndarray
    radius: np.float32
    growth_rate: np.float32
    spring_length_threshold: np.float32
    damping: np.float32
    @staticmethod
    def __new__(
        cls,
        pos,
        vel,
        interaction,
        diffusion_constant=0.0,
        spring_tension=1.0,
        rigidity=2.0,
        spring_length=3.0,
        damping=1.0,
        growth_rate=0.01,
        growth_rate_setter={"mean": 0.01, "std": 0.0},
        spring_length_threshold=6.0,
        spring_length_threshold_setter={"mean": 6.0, "std": 0.0},
        neighbor_reduction=None,
    ): ...
    def __repr__(self) -> str: ...

class RodMechanicsSettings:
    """\
    Contains all settings required to construct :class:`RodMechanics`
    """

    pos: np.ndarray
    vel: np.ndarray
    diffusion_constant: np.float32
    spring_tension: np.float32
    rigidity: np.float32
    spring_length: np.float32
    damping: np.float32

class CellContainer:
    """\
    Resulting type when executing a full simulation
    """

    cells: dict[int, dict[CellIdentifier, tuple[RodAgent, CellIdentifier | None]]]
    parent_map: dict[CellIdentifier, CellIdentifier | None]
    child_map: dict[CellIdentifier, list[CellIdentifier]]
    cell_to_color: dict[CellIdentifier, tuple[np.uint8, np.uint8, np.uint8]]
    color_to_cell: dict[tuple[np.uint8, np.uint8, np.uint8], CellIdentifier]
    path: Path | None

    def get_cells(
        self,
    ) -> dict[int, dict[CellIdentifier, tuple[RodAgent, CellIdentifier | None]]]: ...
    def get_cells_at_iteration(
        self, iteration: int
    ) -> dict[CellIdentifier, tuple[RodAgent, CellIdentifier | None]]: ...
    def get_cell_history(
        self, identifier: CellIdentifier
    ) -> tuple[dict[int, RodAgent], CellIdentifier | None]: ...
    def get_all_iterations(self) -> list[int]: ...
    def get_parent(self, identifier: CellIdentifier) -> CellIdentifier | None: ...
    def get_children(self, identifier: CellIdentifier) -> list[CellIdentifier]: ...
    def get_color(
        self, identifier: CellIdentifier
    ) -> tuple[np.uint8, np.uint8, np.uint8] | None: ...
    def get_cell_from_color(
        self, color: tuple[np.uint8, np.uint8, np.uint8]
    ) -> CellIdentifier | None: ...
    def have_shared_parent(
        self, ident1: CellIdentifier, ident2: CellIdentifier
    ) -> bool: ...
    def get_parent_map(self) -> dict[CellIdentifier, CellIdentifier | None]: ...
    def get_daughter_map(self) -> dict[CellIdentifier, list[CellIdentifier]]: ...
    def get_child_map(self) -> dict[CellIdentifier, list[CellIdentifier]]: ...
    def get_all_identifiers(self) -> list[CellIdentifier]: ...
    def assign_colors_to_cells(
        self,
    ) -> dict[CellIdentifier, tuple[np.uint8, np.uint8, np.uint8]]: ...
    def serialize(self) -> list[int]: ...
    @staticmethod
    def deserialize(bytes: list[int]) -> CellContainer: ...
    @staticmethod
    def load_from_storage(config: Configuration, date: Path | str) -> CellContainer: ...
    def add_ident_divided(
        self, parent: CellIdentifier | None = None
    ) -> CellIdentifier: ...

def generate_positions(
    n_agents: int,
    agent_settings: AgentSettings,
    config: Configuration,
    rng_seed: int = 0,
    dx: tuple[np.float32, np.float32] | tuple[float, float] = (
        np.float32(0.0),
        np.float32(0.0),
    ),
    randomize_positions: np.float32 | float = np.float32(0.0),
    n_vertices: int = 8,
) -> list[np.ndarray]:
    """\
    Creates positions for multiple :class:`RodAgent` which can be used for simulation purposes.

    Args:
        n_agents(int): Number of positions to create
        agent_settings(AgentSettings): See :class:`AgentSettings`
        config(Configuration): See :class:`Configuration`
        rng_seed(int): Seed for generating the random positions
        dx(np.float32): Spacing towards the domain boundary.
    """
    ...

def generate_agents(
    n_agents: int,
    agent_settings: AgentSettings,
    config: Configuration,
    rng_seed: int = 0,
    dx: tuple[np.float32, np.float32] = (np.float32(0.0), np.float32(0.0)),
    randomize_positions: np.float32 = np.float32(0.0),
    n_vertices: int = 8,
) -> list[RodAgent]:
    """\
    Takes exactly the same arguments as the :meth:`generate_positions` function but automatically
    produces agents instead.
    """
    ...

def run_simulation_with_agents(
    config: Configuration, agents: list[RodAgent]
) -> CellContainer:
    """\
    Executes a simulation given a :class:`Configuration` and a list of :class:`RodAgent`.
    """
    ...

def sort_cellular_identifiers(
    identifiers: list[CellIdentifier],
) -> list[CellIdentifier]:
    """\
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

def counter_to_color(counter: int) -> tuple[np.uint8, np.uint8, np.uint8]:
    """\
    Converts an integer counter between 0 and 251^3-1 to an RGB value.
    The reason why 251 was chosen is due to the fact that it is the highest prime number which is
    below 255.
    This will yield a Field of numbers Z/251Z.

    To calculate artistic color values we multiply the counter by 157*163*173 which are three prime
    numbers roughyl in the middle of 255.
    The new numbers can be calculated via

    >>> new_counter = counter * 157 * 163 * 173
    >>> c1, mod = divmod(new_counter, 251**2)
    >>> c2, mod = divmod(mod, 251)
    >>> c3      = mod

    Args:
        counter (int): Counter between 0 and 251^3-1
        artistic (bool): Enables artistic to provide larger differences between single steps instead
            of simple incremental one.

    Returns:
        tuple[int, int, int]: A list with exactly 3 entries containing the calculated color."""
    ...

def color_to_counter(color: tuple[np.uint8, np.uint8, np.uint8]) -> int:
    """\
    Converts a given color back to the counter value.

    The is the inverse of the :function:`counter_to_color` function.
    The formulae can be calculated with the `Extended Euclidean Algorithm
    <https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm>`_.
    The multiplicative inverse (mod 251) of the numbers of 157, 163 and 173 are:

    >>> assert (12590168 * 157) % 251**3 == 1
    >>> assert (13775961 * 163) % 251**3 == 1
    >>> assert (12157008 * 173) % 251**3 == 1

    Thus the formula to calculate the counter from a given color is:

    >>> counter = color[0] * 251**2 + color[1] * 251 + color[2]
    >>> counter = (counter * 12590168) % 251**3
    >>> counter = (counter * 13775961) % 251**3
    >>> counter = (counter * 12157008) % 251**3
    """
    ...

def parents_diff_mask(
    mask1: np.ndarray[tuple[int, int, int], np.dtype[np.uint8]],
    mask2: np.ndarray[tuple[int, int, int], np.dtype[np.uint8]],
    color_to_cell: dict,
    parent_map: dict[CellIdentifier, CellIdentifier | None],
    parent_penalty: np.float32 | float,
) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
    """Calculates the difference between two masks and applies a lower value where one cell is the
    daughter of the other.

    Args:
        mask1(np.ndarray): Mask of segmented cells at one time-point
        mask2(np.ndarray): Mask of segmented cells at other time-point
        cell_container(CellContainer): See :class:`CellContainer`
        parent_penalty(np.float32): Penalty value when one cell is daughter of other.
            Should be between 0 and 1.
    """
    ...

def render_mask_rs(
    cells: dict[CellIdentifier, tuple[RodAgent, CellIdentifier | None]],
    colors: dict[CellIdentifier, tuple[np.uint8, np.uint8, np.uint8]],
    domain_size: tuple[np.float32, np.float32],
    resolution: np.float32,
): ...
