import cr_mech_coli as crm
import numpy as np

def test_convert_pixel_to_length_and_back():
    domain_size = 100.0
    image_resolution = (800, 800)
    p = np.linspace([1.0, 30.0], [12.1, 26.0], 12)
    q = crm.convert_cell_pos_to_pixels(p, domain_size, image_resolution)
    r = crm.convert_pixel_to_position(q, domain_size, image_resolution)
    
    diffs = np.sum((p-r)**2, axis=1)**0.5
    diff_max = np.max(diffs)
    assert diff_max <= domain_size * 1 / image_resolution[0]

def test_convert_length_to_pixel_and_back():
    domain_size = 73.0
    image_resolution = (200, 300)
    p = np.array(np.round(np.linspace([5,3], [150,107], 5)), dtype=int)
    q = crm.convert_pixel_to_position(p, domain_size, image_resolution)
    r = crm.convert_cell_pos_to_pixels(q, domain_size, image_resolution)

    diffs = np.sum(np.array(p-r, dtype=float)**2, axis=1)**0.5
    diff_max = np.max(diffs)
    assert diff_max <= 1
