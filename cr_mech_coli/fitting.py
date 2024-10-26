"""
"""

import numpy as np

from .datatypes import CellContainer, CellIdentifier
from .imaging import color_to_counter
from .cr_mech_coli_rs import parents_diff_mask

def extract_positions(mask: np.ndarray, n_vertices: int = 8) -> list[np.ndarray]:
    pass

def predict_from_mask(mask: np.ndarray, dt: float) -> CellContainer:
    pass

def penalty_area_diff(mask1, mask2) -> float:
    p = (mask1.reshape((-1, 3)) - mask2.reshape((-1, 3)))
    return 1 - np.mean(p == np.array([0, 0, 0,]).T)

def penalty_area_diff_account_parents(
        mask1: np.ndarray,
        mask2: np.ndarray,
        cell_container: CellContainer,
        parent_penalty: float = 0.5,
    ) -> float:
    """
    Calculates the penalty between two masks while accounting for relations between parent and child
    cells.

    Args:
        mask1(np.ndarray): Mask of segmented cells at one time-point
        mask2(np.ndarray): Mask of segmented cells at other time-point
        cell_container(CellContainer): See :class:`CellContainer`
        parent_penalty(float): Penalty value when one cell is daughter of other.
            Should be between 0 and 1.
    Returns:
        np.ndarray: A 2D array containing penalty values between 0 and 1.
    """
    diff_mask = parents_diff_mask(mask1, mask2, cell_container, parent_penalty)
    return np.mean(diff_mask)
