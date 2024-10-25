"""
This module contains functionality to configure and run simulations.

.. list-table:: Cellular History and Relations
    :header-rows: 1

    * - Method
      - Description
    * - :func:`SimResult.get_cells`
      - All simulation snapshots
    * - :func:`SimResult.get_cells_at_iteration`
      - Simulation snapshot at iteration
    * - :func:`SimResult.get_cell_history`
      - History of one particular cell
    * - :func:`SimResult.get_all_identifiers`
      - Get all identifiers of all cells
    * - :func:`SimResult.get_all_identifiers_unsorted`
      - Get all identifiers (unsorted)
    * - :func:`SimResult.get_parent_map`
      - Maps a cell to its parent.
    * - :func:`SimResult.get_child_map`
      - Maps each cell to its children.
    * - :func:`SimResult.get_parent`
      - Get parent of a cell
    * - :func:`SimResult.get_children`
      - Get all children of a cell
    * - :func:`SimResult.cells_are_siblings`
      - Check if two cells have the same parent

"""

from .cr_mech_coli_rs import run_simulation, AgentSettings, RodAgent, RodMechanicsSettings, Configuration, sort_cellular_identifiers, CellIdentifier, SimResult, MorsePotentialF32
