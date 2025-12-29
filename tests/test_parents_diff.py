import cr_mech_coli as crm
import numpy as np


def test_parent_diff1():
    mask1 = np.zeros((100, 100, 3), dtype=np.uint8)
    mask2 = np.zeros((100, 100, 3), dtype=np.uint8)

    mask1[10:30, 20:40] = (1, 1, 1)
    mask1[10:30, 20:40] = (2, 2, 2)

    color_to_cell = {
        (1, 1, 1): crm.CellIdentifier.new_initial(1),
        (2, 2, 2): crm.CellIdentifier.new_initial(2),
    }
    parent_map = {
        crm.CellIdentifier.new_initial(1): None,
        crm.CellIdentifier.new_initial(2): None,
    }

    for parent_penalty in [1.0, 0.5, 0.1, 0.0]:
        diff = crm.parents_diff_mask(
            mask1, mask2, color_to_cell, parent_map, parent_penalty
        )
        assert np.sum(diff) == np.sum(np.all(mask1 != 0, axis=2))


def test_parent_diff2():
    mask1 = np.zeros((200, 200, 3), dtype=np.uint8)
    mask2 = np.zeros((200, 200, 3), dtype=np.uint8)

    mask1[50:150, 50:150] = (1, 0, 0)
    mask2[60:140, 60:140] = (2, 0, 0)

    color_to_cell = {
        (1, 0, 0): crm.CellIdentifier.new_initial(1),
        (2, 0, 0): crm.CellIdentifier.new_initial(2),
    }
    parent_map = {
        crm.CellIdentifier.new_initial(1): crm.CellIdentifier.new_initial(2),
        crm.CellIdentifier.new_initial(2): None,
    }

    for parent_penalty in [1.0, 0.5, 0.1, 0.0]:
        diff = crm.parents_diff_mask(
            mask1, mask2, color_to_cell, parent_map, parent_penalty
        )
        filt = np.any(mask2 != 0, axis=2)
        a1 = np.sum(np.any(mask1 != 0, axis=2) * ~filt)
        a2 = np.sum(filt)
        b = np.sum(diff)
        print(a1, a2, b)
        assert b == a1 + a2 * parent_penalty
