import numpy as np

import cr_mech_coli as crm

def get_color_mappings(
    container: crm.CellContainer,
    masks_data: list[np.ndarray[tuple[int, int], np.dtype[np.uint8]]],
    iterations_data: list[int],
    positions_all: list[np.ndarray[tuple[int, int, int], np.dtype[np.float32]]],
) -> tuple[
    dict[int, dict[np.uint8, crm.CellIdentifier]],
    dict[tuple[np.uint8, np.uint8, np.uint8], crm.CellIdentifier],
    dict[crm.CellIdentifier, crm.CellIdentifier | None],
]: ...
