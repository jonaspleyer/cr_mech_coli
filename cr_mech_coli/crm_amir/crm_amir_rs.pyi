from cr_mech_coli import RodAgent
import numpy as np

class FixedRod:
    agent: RodAgent
    domain_size: float
    block_size: float

class Parameters:
    domain_size: float
    block_size: float
    drag_force: float
    t_max: float
    save_interval: float
    dt: float
    rod_length: float
    rod_rigiditiy: float
    spring_tension: float
    growth_rate: float
    damping: float
    n_vertices: int
    progressbar: str | None

    def __new__(cls) -> Parameters: ...

def run_sim(
    parameters: Parameters, pos: np.ndarray | None = None
) -> list[tuple[int, FixedRod]]: ...
