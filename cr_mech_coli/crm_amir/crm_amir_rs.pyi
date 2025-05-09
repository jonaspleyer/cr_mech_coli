from cr_mech_coli import RodAgent

class FixedRod:
    agent: RodAgent
    domain_size: float
    block_size: float

class Parameters:
    domain_size: float
    block_size: float
    t_max: float
    save_interval: float
    dt: float
    rod_length: float
    rod_rigiditiy: float
    n_vertices: int

    def __new__(cls) -> Parameters: ...

def run_sim(parameters: Parameters): ...
