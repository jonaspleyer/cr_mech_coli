"""
"""

import numpy as np

from .datatypes import CellContainer, CellIdentifier
from .imaging import color_to_counter

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
        parent_penalty: float = 0.25,
    ) -> float:
    m1 = mask1.reshape((-1, 3))
    m2 = mask2.reshape((-1, 3))
    penalty = 0
    for i in range(m1.shape[0]):
        x1 = m1[i,:]
        x2 = m2[i,:]
        if np.all(x1 != x2):
            c1 = cell_container.get_cell_from_color(x1)
            c2 = cell_container.get_cell_from_color(x2)
            if c1 is not None and c2 is not None:
                pc1 = cell_container.get_parent(c1)
                pc2 = cell_container.get_parent(c2)
                if pc1 == pc2:
                    penalty += parent_penalty
                else:
                    penalty += 1
            else:
                penalty += 1

    return penalty / m1.shape[0]

