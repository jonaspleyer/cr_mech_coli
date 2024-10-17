from .cr_mech_coli_rs import sort_cellular_identifiers

"""
This module contains functionality to configure and run simulations.
"""

def extract_all_identifiers(cells_at_iterations: dict) -> set:
    """Extracts all identifiers of the given simulation data and returns them as a set.

    Args:
        cells_at_iterations (dict):
            A dictionary of which given an iteration returns the cells at this simulation step as
            produced by the `run_simulation` function.

    Returns:
        set: All identifiers which were present during the simulation.
    """
    return set(cells_at_iterations.keys())
