"""
This module provides functionality around fitting the :ref:`model` to given data.

.. list-table:: Compare Masks
    :header-rows: 0
    :widths: 40 60

    * - :func:`area_diff_mask`
      - Computes a 2D array where the two masks differ.
    * - :func:`penalty_area_diff`
      - Calculates the penalty based on difference in colors.
    * - :func:`parents_diff_mask`
      - Computes a 2D penalty array which accounts if cells are related.
    * - :func:`penalty_area_diff_account_parents`
      - Uses the :func:`parents_diff_mask` to calculate the associated penatly.

.. list-table:: Work with Masks and Cell Positions
    :header-rows: 0
    :widths: 40 60

    * - :func:`extract_positions`
      - Extracts a list of position from a given mask.
    * - :func:`convert_pixel_to_position`
      - Converts positions in pixel units to units of length (typically µm).
    * - :func:`convert_cell_pos_to_pixels`
      - Converts positions in length units (typically µm) to pixel units.
"""

import numpy as np
import skimage as sk

from .datatypes import CellContainer, CellIdentifier
from .imaging import color_to_counter
from .cr_mech_coli import parents_diff_mask


def points_along_polygon(
    polygon: list[np.ndarray] | np.ndarray, n_vertices: int = 8
) -> np.ndarray:
    """
    Returns evenly-spaced points along the given polygon.
    The initial and final point are always included.

    Args:
        polygon(list[np.ndarray] | np.ndarray: Ordered points which make up the polygon.
        n_vertices(int): Number of vertices which should be extracted.
    Returns:
        np.ndarray: Array containing all extracted points (along the 0th axis).
    """
    polygon = np.array(polygon, dtype=np.float32)

    # Calculate the total length
    length_segments = np.sqrt(np.sum((polygon[1:] - polygon[:-1]) ** 2, axis=1))
    length_segments_increasing = np.cumsum([0, *length_segments])
    length_total = length_segments_increasing[-1]
    dx = length_total / (n_vertices - 1)

    points = [polygon[0]]
    for i in range(1, n_vertices - 1):
        diffs = i * dx - length_segments_increasing
        j = max(int(np.argmin(diffs > 0)) - 1, 0)
        p1 = polygon[j]
        p2 = polygon[j + 1]
        t = diffs[j] / length_segments[j]
        p_new = p1 * (1 - t) + p2 * t
        points.append(p_new)
    points.append(polygon[-1])
    return np.array(points, dtype=np.float32)


def extract_positions(
    mask: np.ndarray,
    n_vertices: int = 8,
    skel_method="lee",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts positions from a mask for each sub-mask associated to a single cell.
    To read more about the used methods, visit the :ref:`Fitting-Methods` page.

    Args:
        mask(np.ndarray): Array of shape :code:`(D1, D2, 3)`, `(D1, D2, 1)` or `(D1, D2)` containing
            pixel values of a mask for multiple cells.
        n_vertices(int): Number of vertices which should be extracted from each given cell-mask.
    Returns:
        tuple[np.ndarray]: A tuple containing
        1. Array :code:`(n_agents, n_vertices, 2)` containing the positions of the cells
        2. Array :code:`(n_agents)` containing the lengths of the rods
        3. Array :code:`(n_agents)` containing rough approximate radii of the cells
    """
    from .cr_mech_coli import _sort_points

    # First determine the number of unique identifiers
    if len(mask.shape) == 3:
        assert mask.shape[2] == 3 or mask.shape[2] == 1
        m = mask.reshape((-1, mask.shape[2]))
    elif len(mask.shape) == 2:
        m = mask.reshape(-1)
    else:
        raise ValueError(
            "We only support masks with shapes (n, m), (n, m, 1) or (n, m, 3)"
        )
    colors = filter(lambda x: np.sum(x) != 0, np.unique(m, axis=0))

    if len(mask.shape) > 2:
        cell_masks = [np.all(mask == c, axis=2) for c in colors]
    else:
        cell_masks = [mask == c for c in colors]
    skeleton_points = [
        _sort_points(sk.morphology.skeletonize(m, method=skel_method))
        for m in cell_masks
    ]
    polys = [sk.measure.approximate_polygon(sp, 1) for sp in skeleton_points]
    points = np.array(
        [np.roll(points_along_polygon(p, n_vertices), 1, axis=1) for p in polys]
        dtype=np.float32,
    )

    lengths = np.sum(np.linalg.norm(points[:, 1:] - points[:, :-1], axis=2), axis=1)
    areas = np.array([np.sum(c) for c in cell_masks], dtype=np.float32)
    radii = lengths / np.pi * (np.sqrt(1 + np.pi * areas / lengths**2) - 1)
    return points, lengths, radii


def area_diff_mask(mask1, mask2) -> np.ndarray:
    """
    Calculates a 2D array with entries 1 whenever colors differ and 0 if not.

    Args:
        mask1(np.ndarray): Mask of segmented cells at one time-point
        mask2(np.ndarray): Mask of segmented cells at other time-point
    Returns:
        np.ndarray: A 2D array with entries of value 1 where a difference was calculated.
    """
    s = mask1.shape
    p = mask1.reshape((-1, 3)) - mask2.reshape((-1, 3))
    return (p != np.array([0, 0, 0]).T).reshape(s)[:, :, 0]


def penalty_area_diff(mask1, mask2) -> float:
    """
    Calculates the penalty between two masks based on differences in color values (See also:
    :func:`area_diff_mask`).

    Args:
        mask1(np.ndarray): Mask of segmented cells at one time-point
        mask2(np.ndarray): Mask of segmented cells at other time-point
    Returns:
        float: The penalty
    """
    p = mask1.reshape((-1, 3)) - mask2.reshape((-1, 3))
    return np.mean(
        p
        != np.array(
            [
                0,
                0,
                0,
            ]
        ).T
    )


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


def convert_cell_pos_to_pixels(
    cell_pos: np.ndarray,
    domain_size: np.ndarray | tuple[float, float] | float,
    image_resolution: tuple[int, int] | np.ndarray,
):
    """
    Converts the position of a cell (collection of vertices) from length units (typically µm) to
    pixels.

    .. warning::
        This function performs only an approximate inverse to the :func:`convert_pixel_to_position`
        function.
        When converting from floating-point values to pixels and back rounding errors will be
        introcued.

    Args:
        cell_pos(np.ndarray): Array of shape (N,2) containing the position of the cells vertices.
        domain_size(float): The overall edge length of the domain (typically in µm).
        image_resolution(tuple[int, int] | np.ndarray): A tuple containing the resolution of the
            image.
            Typically, the values for width and height are identical.
    Returns:
        np.ndarray: The converted position of the cell.
    """
    if type(domain_size) is float:
        domain_size = np.array([domain_size] * 2, dtype=float)
    domain_size = np.array(domain_size)

    dl = 2**0.5 * domain_size
    domain_pixels = np.array(image_resolution, dtype=float)
    pixel_per_length = domain_pixels / dl

    # Shift coordinates to center
    p = 0.5 * domain_size - cell_pos[:, :2]
    # Scale with conversion between pixels and length
    p *= np.array([-1, 1]) * pixel_per_length
    # Shift coordinate system again
    p += 0.5 * domain_pixels
    # Round to plot in image
    p = np.array(np.round(p), dtype=int)
    return p


def convert_pixel_to_position(
    pos_pixel: np.ndarray,
    domain_size: np.ndarray | tuple[float, float] | float,
    image_resolution: tuple[float, float],
):
    """
    Contains identical arguments as the :func:`convert_cell_pos_to_pixels` function and performs the
    approximate inverse operation.

    .. warning::
        This function performs only an approximate inverse to the :func:`convert_cell_pos_to_pixels`
        function.
        When converting from floating-point values to pixels and back rounding errors will be
        introcued.

    Returns:
        np.ndarray: The converted position of the cell.
    """
    # Convert to numpy array
    if type(domain_size) is float:
        domain_size = np.array([domain_size] * 2, dtype=float)
    domain_size = np.array(domain_size, dtype=float)

    dl = 2**0.5 * domain_size
    domain_pixels = np.array(image_resolution, dtype=float)
    pixel_per_length = domain_pixels / dl

    p = np.array(pos_pixel, dtype=float)
    p -= 0.5 * domain_pixels
    p = p * np.array([-1, 1]) / pixel_per_length
    p = 0.5 * domain_size - p
    return p
