"""
This module contains storage solutions for information about cellular history and their relations.

.. list-table:: Cellular History and Relations
    :header-rows: 1

    * - Method
      - Description
    * - :func:`CellContainer.get_cells`
      - All simulation snapshots
    * - :func:`CellContainer.get_cells_at_iteration`
      - Simulation snapshot at iteration
    * - :func:`CellContainer.get_cell_history`
      - History of one particular cell
    * - :func:`CellContainer.get_all_identifiers`
      - Get all identifiers of all cells
    * - :func:`CellContainer.get_all_identifiers_unsorted`
      - Get all identifiers (unsorted)
    * - :func:`CellContainer.get_parent_map`
      - Maps a cell to its parent.
    * - :func:`CellContainer.get_child_map`
      - Maps each cell to its children.
    * - :func:`CellContainer.get_parent`
      - Get parent of a cell
    * - :func:`CellContainer.get_children`
      - Get all children of a cell
    * - :func:`CellContainer.cells_are_siblings`
      - Check if two cells have the same parent

"""

from .cr_mech_coli import CellContainer, CellIdentifier
