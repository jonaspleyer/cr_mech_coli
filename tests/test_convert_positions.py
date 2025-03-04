import cr_mech_coli as crm
import numpy as np


def check_diff_float(p, r, domain_size, image_resolution):
    if type(domain_size) is float:
        domain_size = [domain_size] * 2
    domain_size = np.array(domain_size)
    diffs = np.abs(p - r)
    assert np.all(diffs <= domain_size / image_resolution[::-1])


def check_diff_pixel(p, r):
    assert np.max(np.abs(p - r)) == 0


def test_convert_pixel_to_length_and_back():
    domain_size = 100.0
    image_resolution = (800, 800)
    p = np.linspace([1.0, 30.0], [12.1, 26.0], 12)
    q = crm.convert_cell_pos_to_pixels(p, domain_size, image_resolution)
    r = crm.convert_pixel_to_position(q, domain_size, image_resolution)
    check_diff_float(p, r, domain_size, image_resolution)


def test_convert_length_to_pixel_and_back():
    domain_size = 73.0
    image_resolution = (200, 300)
    p = np.array(np.round(np.linspace([5, 3], [150, 107], 5)), dtype=int)
    q = crm.convert_pixel_to_position(p, domain_size, image_resolution)
    r = crm.convert_cell_pos_to_pixels(q, domain_size, image_resolution)
    check_diff_pixel(p, r)


def test_convert_non_square_domain_length_to_pixel_and_back():
    domain_size = (100.0, 50.0)
    image_resolution = (100, 600)
    p = np.linspace([88.3, 45.0], [63.7, 23.1], 20)
    q = crm.convert_cell_pos_to_pixels(p, domain_size, image_resolution)
    r = crm.convert_pixel_to_position(q, domain_size, image_resolution)
    check_diff_float(p, r, domain_size, image_resolution)


def test_convert_non_square_domain_pixel_to_length_and_back():
    domain_size = (800_038.0, 739.4)
    image_resolution = (400, 300)
    p = np.array(np.round(np.linspace([200, 146], [163, 87], 30)), dtype=int)
    q = crm.convert_pixel_to_position(p, domain_size, image_resolution)
    r = crm.convert_cell_pos_to_pixels(q, domain_size, image_resolution)
    check_diff_pixel(p, r)


def test_convert_non_square_domain_domain_identity():
    domain_size = (100, 200)
    image_resolution = (200, 100)
    p = np.linspace([10.0, 10.0], [40.0, 40.0])
    q = crm.convert_cell_pos_to_pixels(p, domain_size, image_resolution)
    q[:, 0] = image_resolution[0] - q[:, 0]
    check_diff_float(p, q[:, ::-1], domain_size, image_resolution)


def test_convert_non_square_pixel_domain_identity():
    domain_size = (450.0, 150.0)
    image_resolution = (150, 450)
    p = np.array(np.round(np.linspace([15, 10], [100, 350])), dtype=int)
    q = crm.convert_pixel_to_position(p, domain_size, image_resolution)
    q[:, 1] = domain_size[1] - q[:, 1]
    check_diff_pixel(p, q[:, ::-1])


def test_convert_domain_pixel_slope():
    domain_size = (300, 300)
    image_resolution = (1000, 600)

    # This will have slope 1 in domain units
    p = np.linspace([0, 0], domain_size, 10, dtype=float)
    q = crm.convert_cell_pos_to_pixels(p, domain_size, image_resolution)

    # Calculate ratios of domain and image_resolution
    ratio_domain = p / np.array(domain_size)
    q[:, 1] = image_resolution[1] - q[:, 1]
    ratio_image = q[::-1] / np.array(image_resolution)
    check_diff_float(ratio_domain, ratio_image, domain_size, image_resolution)


def test_convert_pixel_domain_slope():
    domain_size = (10_000, 300_000)
    image_resolution = (1200, 800)

    # This will have slope -1
    q = np.linspace([0, image_resolution[1]], [image_resolution[0], 0], 10, dtype=int)
    p = crm.convert_pixel_to_position(q, domain_size, image_resolution)

    ratio_domain = p / np.array(domain_size)
    q[:, 1] = image_resolution[1] - q[:, 1]
    ratio_image = q / np.array(image_resolution)
    check_diff_float(ratio_domain, ratio_image, domain_size, image_resolution)
