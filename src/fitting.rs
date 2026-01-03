use pyo3::prelude::*;

/// Checks if both arrays have identical shape and are non-empty
macro_rules! check_shape_identical_nonempty(
    ($a1:ident, $a2:ident) => {
        if $a1.shape() != $a2.shape() || $a1.shape().len() == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Masks need to have matching nonempty shapes. Got shapes: {:?} {:?}",
                $a1.shape(),
                $a2.shape()
            )));
        }
    };
);

/// Simplify conversion of generic error messages to pyo3 errors
macro_rules! new_error (
    ($error_kind:ident, $($message:tt),*) => {
        pyo3::exceptions:: $error_kind ::new_err(format!($($message),*))
    };
);

/// Calculates the difference between two masks and applies a lower value where one cell is the
/// daughter of the other.
///
/// Args:
///     mask1(np.ndarray): Mask of segmented cells at one time-point
///     mask2(np.ndarray): Mask of segmented cells at other time-point
///     color_to_cell(dict): Maps colors of type `tuple[u8, u8, u8]` to :class:`CellIdentifier`
///     parent_map(dict): Maps cellidentifiers to their (optional) parent
///     cell_container(CellContainer): See :class:`CellContainer`
///     parent_penalty(float): Penalty value when one cell is daughter of other.
///         Should be between 0 and 1.
#[pyfunction]
#[pyo3(signature = (mask1, mask2, color_to_cell, parent_map, parent_penalty = 0.5))]
pub fn parents_diff_mask<'py>(
    py: Python<'py>,
    mask1: numpy::PyReadonlyArray3<'py, u8>,
    mask2: numpy::PyReadonlyArray3<'py, u8>,
    color_to_cell: std::collections::BTreeMap<(u8, u8, u8), crate::CellIdentifier>,
    parent_map: std::collections::BTreeMap<crate::CellIdentifier, Option<crate::CellIdentifier>>,
    parent_penalty: f32,
) -> pyo3::PyResult<Bound<'py, numpy::PyArray2<f32>>> {
    use numpy::*;
    let m1 = mask1.as_array();
    let m2 = mask2.as_array();
    check_shape_identical_nonempty!(m1, m2);
    let s = m1.shape();
    let new_shape = [s[0] * s[1], s[2]];
    let m1 = m1
        .to_shape(new_shape)
        .map_err(|e| new_error!(PyValueError, "{e}"))?;
    let m2 = m2
        .to_shape(new_shape)
        .map_err(|e| new_error!(PyValueError, "{e}"))?;
    let diff_mask = numpy::ndarray::Array1::<f32>::from_iter(
        m1.outer_iter()
            .zip(m2.outer_iter())
            .map(|(c1, c2)| {
                if c1 != c2 && c1.sum() != 0 && c2.sum() != 0 {
                    let c1 = (c1[0], c1[1], c1[2]);
                    let c2 = (c2[0], c2[1], c2[2]);

                    let i1 = color_to_cell.get(&c1).ok_or(new_error!(
                        PyKeyError,
                        "could not find color {:?}",
                        c1
                    ))?;
                    let i2 = color_to_cell.get(&c2).ok_or(new_error!(
                        PyKeyError,
                        "could not find color {:?}",
                        c2
                    ))?;

                    // Check if one is the parent of the other
                    let p1 = parent_map.get(i1).ok_or(new_error!(
                        PyKeyError,
                        "could not find cell {:?}",
                        i1
                    ))?;
                    let p2 = parent_map.get(i2).ok_or(new_error!(
                        PyKeyError,
                        "could not find cell {:?}",
                        i2
                    ))?;

                    if Some(i1) == p2.as_ref() || Some(i2) == p1.as_ref() {
                        return Ok(parent_penalty);
                    }
                    Ok(1.0)
                } else if c1 != c2 {
                    Ok(1.0)
                } else {
                    Ok(0.0)
                }
            })
            .collect::<pyo3::PyResult<Vec<f32>>>()?,
    );
    let diff_mask = diff_mask
        .to_shape([s[0], s[1]])
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
    Ok(diff_mask.to_pyarray(py))
}

struct Bbox {
    xmin: f32,
    ymin: f32,
    xmax: f32,
    ymax: f32,
}

fn calculate_bounding_box_with_padding(vertices: &ndarray::ArrayView2<f32>, r: f32) -> Bbox {
    let mut xmin = -f32::INFINITY;
    let mut ymin = -f32::INFINITY;
    let mut xmax = f32::INFINITY;
    let mut ymax = f32::INFINITY;

    for p in vertices.rows().into_iter() {
        xmin = xmin.min(p[0]);
        ymin = xmin.min(p[1]);

        xmax = xmax.max(p[0]);
        ymax = ymax.max(p[1]);
    }

    xmax += r;
    xmin -= r;
    ymax += r;
    ymin -= r;

    Bbox {
        xmin,
        xmax,
        ymin,
        ymax,
    }
}

fn bounding_boxes_intersect_with_padding(bbox1: &Bbox, bbox2: &Bbox) -> bool {
    let cond1 = bbox1.xmin < bbox2.xmax;
    let cond2 = bbox1.xmax > bbox2.xmin;
    let cond3 = bbox1.ymax > bbox2.ymin;
    let cond4 = bbox1.ymin < bbox2.ymax;
    cond1 && cond2 && cond3 && cond4
}

/// Calculates the approximate overlap of agents at a current simulation state
#[pyfunction]
#[pyo3(signature = (
    positions,
    radii,
    delta_angle = core::f32::consts::FRAC_PI_4,
    epsilon = 0.01,
))]
pub fn overlap(
    positions: numpy::PyReadonlyArray3<f32>,
    radii: numpy::PyReadonlyArray1<f32>,
    delta_angle: f32,
    epsilon: f32,
) -> PyResult<f32> {
    // Calculate overlaps between agents as 2D volume
    let positions = positions.as_array();
    let radii = radii.as_array();

    let mut polygons = Vec::with_capacity(radii.len());
    for n in 0..radii.len() {
        let p = positions.slice(ndarray::s![n, .., ..]);
        let r = radii[n];

        let polygon = calcualte_polygon_hull(&p, r, delta_angle, epsilon)?;
        polygons.push(polygon);
    }

    let mut total_area = 0.0;
    for (n, p1) in polygons.iter().enumerate() {
        let x1 = positions.slice(ndarray::s![n, .., ..]);
        let r1 = radii[n];
        let bbox1 = calculate_bounding_box_with_padding(&x1, r1);

        for (m, p2) in polygons.iter().enumerate().skip(n + 1) {
            let x2 = positions.slice(ndarray::s![m, .., ..]);
            let r2 = radii[m];
            let bbox2 = calculate_bounding_box_with_padding(&x2, r2);

            if bounding_boxes_intersect_with_padding(&bbox1, &bbox2) {
                // Calculate overlap between polygons
                use geo::{Area, BooleanOps};
                let intersection = p1.intersection(p2);
                total_area += intersection.unsigned_area();
            }
        }
    }

    Ok(total_area)
}

// Angle in radians [0,2pi]
fn generate_coordinates_sphere(
    p: &ndarray::ArrayView1<f32>,
    r: f32,
    angle_start: f32,
    angle_end: f32,
    delta_angle: f32,
) -> Vec<geo::Coord<f32>> {
    let n_resolution = get_n_resolution(angle_start, angle_end, delta_angle);
    let dangle = (angle_end - angle_start).abs() / (n_resolution as f32 + 1.0);
    (1..n_resolution + 1)
        .map(|n| {
            let angle = angle_start + dangle * n as f32;
            geo::coord! { x: p[0] + angle.cos() * r, y: p[1] + angle.sin() * r }
        })
        .collect()
}

/// Returns `[c0, c1, c2, c3]` defined by
///
/// c0 ------------ c1
/// |               |
/// p1   --dir->    p2
/// |               |
/// c3 ------------ c2
fn generate_coordinates_rectangle(
    p1: &ndarray::ArrayView1<f32>,
    p2: &ndarray::ArrayView1<f32>,
    r: f32,
) -> [geo::Coord<f32>; 4] {
    let norm = ((p2[0] - p1[0]).powi(2) + (p2[1] - p1[1]).powi(2)).sqrt();
    let dir = [(p2[0] - p1[0]) / norm, (p2[1] - p1[1]) / norm];

    let c0 = geo::coord! { x: p1[0] - r*dir[1], y: p1[1] + r*dir[0] };
    let c1 = geo::coord! { x: p2[0] - r*dir[1], y: p2[1] + r*dir[0] };
    let c2 = geo::coord! { x: p2[0] + r*dir[1], y: p2[1] - r*dir[0] };
    let c3 = geo::coord! { x: p1[0] + r*dir[1], y: p1[1] - r*dir[0] };

    [c0, c1, c2, c3]
}

fn angle_to_x_axis(dir: &impl core::ops::Index<usize, Output = f32>) -> f32 {
    nalgebra::RealField::atan2(dir[1], dir[0]).rem_euclid(2.0 * core::f32::consts::PI)
}

fn angles_between(dir1: &[f32; 2], dir2: &[f32; 2]) -> (f32, f32) {
    let dir1 = dir1.as_ref();
    let dir2 = dir2.as_ref();
    let perp = dir1[0] * dir2[1] - dir1[1] * dir2[0];
    let dot = dir1[0] * dir2[0] + dir1[1] * dir2[1];

    let angle_lh = nalgebra::RealField::atan2(perp, dot).rem_euclid(2.0 * core::f32::consts::PI);
    let angle_rh = nalgebra::RealField::atan2(-perp, dot).rem_euclid(2.0 * core::f32::consts::PI);

    (angle_lh, angle_rh)
}

fn get_n_resolution(angle_start: f32, angle_end: f32, delta_angle: f32) -> usize {
    use approx::AbsDiffEq;
    let dangle = (angle_start - angle_end).abs();
    // let angle_frac = core::f32::consts::PI * 2.0 / n_resolution as f32;
    let n_resolution = (dangle / delta_angle).max(1.0);
    if (n_resolution % 1.0).abs_diff_eq(&0.0, 0.001) {
        (n_resolution - 2.0) as usize
    } else {
        (n_resolution - 1.0) as usize
    }
    .max(1)
}

fn determine_spheroid_coordinates_between_rectangles(
    p1: &ndarray::ArrayView1<f32>,
    p2: &ndarray::ArrayView1<f32>,
    p3: &ndarray::ArrayView1<f32>,
    r: f32,
    delta_angle: f32,
) -> (bool, Vec<geo::Coord<f32>>) {
    // Insert coordinates for connecting spheroid
    // Case 1: ---\
    //             \
    //
    //             /
    // Case 2: ---/

    // Determine the angles between the direction vertices
    let dir1 = [p2[0] - p1[0], p2[1] - p1[1]];
    let dir2 = [p3[0] - p2[0], p3[1] - p2[1]];

    let (angle_lh, angle_rh) = angles_between(&dir1, &dir2);

    let [_, c1, c2, _] = generate_coordinates_rectangle(p1, p2, r);
    let [d0, _, _, d3] = generate_coordinates_rectangle(p2, p3, r);

    // Case 1
    let (is_left, angle_start, angle_end) = if angle_lh > angle_rh {
        let x1 = ndarray::array![d0.x - p2[0], d0.y - p2[1]];
        let x2 = ndarray::array![c1.x - p2[0], c1.y - p2[1]];
        let angle_start = angle_to_x_axis(&x1.view());
        let angle_end = angle_to_x_axis(&x2.view());
        (true, angle_start, angle_end)
    }
    // Case 2
    else {
        let x1 = ndarray::array![d3.x - p2[0], d3.y - p2[1]];
        let x2 = ndarray::array![c2.x - p2[0], c2.y - p2[1]];
        let angle_start = angle_to_x_axis(&x2.view());
        let angle_end = angle_to_x_axis(&x1.view());
        (false, angle_start, angle_end)
    };

    (
        is_left,
        generate_coordinates_sphere(p2, r, angle_start, angle_end, delta_angle),
    )
}

/// S   c0___
/// P  /
/// H /
/// E |
/// R \
/// E  \c3___
fn get_coordinates_at_tip(
    p1: &ndarray::ArrayView1<f32>,
    p2: &ndarray::ArrayView1<f32>,
    r: f32,
    delta_angle: f32,
) -> (Vec<geo::Coord<f32>>, geo::Coord<f32>) {
    let norm = ((p2[0] - p1[0]).powi(2) + (p2[1] - p1[1]).powi(2)).sqrt();
    let dir = (p2.to_owned() - p1) / norm;
    let angle_start = angle_to_x_axis(&[-dir[1], dir[0]]);
    let angle_end = angle_to_x_axis(&[dir[1], -dir[0]]);

    let mut coordinates1 = Vec::new();

    let [c0, _, _, c3] = generate_coordinates_rectangle(p1, &p2.view(), r);
    coordinates1.extend(generate_coordinates_sphere(
        p1,
        r,
        angle_start,
        angle_end,
        delta_angle,
    ));
    coordinates1.push(c3);

    (coordinates1, c0)
}

fn calcualte_polygon_hull(
    p: &ndarray::ArrayView2<f32>,
    r: f32,
    delta_angle: f32,
    epsilon: f32,
) -> Result<geo::Polygon<f32>, cellular_raza::prelude::SimulationError> {
    let mut coordinates_fw = Vec::new();
    let mut coordinates_bw = Vec::new();

    // Forward pass

    // Add sphere-like coordinates for starting tip
    let p1 = p.slice(ndarray::s![0, ..]);
    let p2 = p.slice(ndarray::s![1i32, ..]);
    // let x1 = geo::coord! { x: p1[0] - r * dir[1], y: p1[1] + r * dir[0] };
    // let x2 = geo::coord! { x: p1[0] + r * dir[1], y: p1[1] - r * dir[0] };

    // let angle_end = nalgebra::RealField::atan2(x2.y, x2.x) % (2.0 * core::f32::consts::PI);

    // Assembly the polygon in this order:
    //   x--------------coordinates_fw ---------->
    //  /                                         \
    // /                                           \
    // |                                           |
    // |                                           |
    // \                                           /
    //  \                                         /
    //    ------------- coordinates_bw --------> x

    let (coords1, coord_fin) = get_coordinates_at_tip(&p1.view(), &p2.view(), r, delta_angle);
    coordinates_bw.extend(coords1);
    coordinates_fw.push(coord_fin);

    for n in 2..p.shape()[0] {
        let p1 = p.slice(ndarray::s![n - 2, ..]);
        let p2 = p.slice(ndarray::s![n - 1, ..]);
        let p3 = p.slice(ndarray::s![n, ..]);

        // Insert coordiantes for rectangles
        let [c0, c1, c2, c3] = generate_coordinates_rectangle(&p1, &p2, r);
        let [d0, d1, d2, d3] = generate_coordinates_rectangle(&p2, &p3, r);

        let (is_left, coordinates) =
            determine_spheroid_coordinates_between_rectangles(&p1, &p2, &p3, r, delta_angle);

        use approx::AbsDiffEq;
        if is_left {
            //   c1
            // _____        COORDINATES_FW
            //       .
            // d3     \ d0
            // __\_c2  \
            //    \     \
            //     \
            // COORDINATES_BW
            coordinates_fw.push(c1);
            // We need to reverse the iterator since it produces
            // coordinates in left-handed circular direction.
            coordinates_fw.extend(coordinates.into_iter().rev());
            coordinates_fw.push(d0);
            // Calculate intersection point of [c3 - c2] and [d3 - d2]
            // and add this point
            let line1 = geo::Line { start: c3, end: c2 };
            let line2 = geo::Line { start: d3, end: d2 };
            match geo::line_intersection::line_intersection(line1, line2) {
                Some(geo::LineIntersection::SinglePoint {
                    intersection,
                    is_proper: _,
                }) => coordinates_bw.push(intersection),
                _ => {
                    if [c2.x, c2.y].abs_diff_eq(&[d3.x, d3.y], epsilon) {
                        coordinates_bw.push(c2);
                    } else {
                        return Err(cellular_raza::prelude::CalcError(format!(
                            "lines should be intersecting: {c3:?}--{c2:?}; {d3:?}--{d2:?}"
                        ))
                        .into());
                    }
                }
            }
        } else {
            // Same as in the other condition but reversed sides
            coordinates_bw.push(c2);
            coordinates_bw.extend(coordinates.into_iter());
            coordinates_bw.push(d3);
            let line1 = geo::Line { start: c0, end: c1 };
            let line2 = geo::Line { start: d0, end: d1 };
            match geo::line_intersection::line_intersection(line1, line2) {
                Some(geo::LineIntersection::SinglePoint {
                    intersection,
                    is_proper: _,
                }) => coordinates_fw.push(intersection),
                _ => {
                    if [c1.x, c1.y].abs_diff_eq(&[d0.x, d0.y], epsilon) {
                        coordinates_bw.push(c1);
                    } else {
                        return Err(cellular_raza::prelude::CalcError(format!(
                            "lines should be intersecting: {c3:?}--{c2:?}; {d3:?}--{d2:?}"
                        ))
                        .into());
                    }
                }
            }
        }
    }

    let pfin2 = p.slice(ndarray::s![-2, ..]);
    let pfin = p.slice(ndarray::s![-1, ..]);
    let (coords1, coord_fin) = get_coordinates_at_tip(&pfin, &pfin2, r, delta_angle);
    coordinates_bw.push(coord_fin);
    coordinates_fw.extend(coords1.into_iter().rev());

    let mut coordinates = coordinates_fw;
    coordinates.extend(coordinates_bw.into_iter().rev());

    Ok(geo::Polygon::<f32>::new(
        geo::LineString::new(coordinates),
        vec![],
    ))
}

/// Helper function to sort points from a skeletonization in order.
#[pyfunction]
pub fn _sort_points<'py>(
    py: Python<'py>,
    skeleton: numpy::PyReadonlyArray2<'py, bool>,
) -> pyo3::PyResult<Bound<'py, numpy::PyArray2<isize>>> {
    use core::ops::{AddAssign, MulAssign};
    use numpy::ndarray::prelude::*;
    use numpy::ToPyArray;
    let skeleton = skeleton.as_array().mapv(|x| x as u8);
    let mut neighbors = Array2::<u8>::zeros(skeleton.dim());

    //   x
    // x x x
    //   x
    neighbors
        .slice_mut(s![1.., ..])
        .add_assign(&skeleton.slice(s![..-1, ..]));
    neighbors
        .slice_mut(s![..-1, ..])
        .add_assign(&skeleton.slice(s![1.., ..]));
    neighbors
        .slice_mut(s![.., 1..])
        .add_assign(&skeleton.slice(s![.., ..-1]));
    neighbors
        .slice_mut(s![.., ..-1])
        .add_assign(&skeleton.slice(s![.., 1..]));

    // Corners
    // x   x
    //   x
    // x   x
    neighbors
        .slice_mut(s![1.., 1..])
        .add_assign(&skeleton.slice(s![..-1, ..-1]));
    neighbors
        .slice_mut(s![..-1, 1..])
        .add_assign(&skeleton.slice(s![1.., ..-1]));
    neighbors
        .slice_mut(s![1.., ..-1])
        .add_assign(&skeleton.slice(s![..-1, 1..]));
    neighbors
        .slice_mut(s![..-1, ..-1])
        .add_assign(&skeleton.slice(s![1.., 1..]));

    neighbors.mul_assign(&skeleton);

    let mut x = Vec::new();
    let mut y = Vec::new();
    let (mut e1, mut e2) = (None, None);
    for i in 0..neighbors.dim().0 {
        for j in 0..neighbors.dim().1 {
            if neighbors[(i, j)] == 1 {
                if e1.is_none() {
                    e1 = Some(numpy::ndarray::array![i as isize, j as isize]);
                } else if e2.is_none() {
                    e2 = Some(numpy::ndarray::array![i as isize, j as isize]);
                } else {
                    // If we find more points which are matching we return an error
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Detected more than 2 endpoints after skeletonization",
                    ));
                }
            }

            // This collects all indices where the skeleton lives
            if skeleton[(i, j)] == 1 {
                x.push(i as isize);
                y.push(j as isize);
            }
        }
    }
    if let (Some(e1), Some(e2)) = (e1, e2) {
        // Pre-Sort the points
        let n_unique_x = std::collections::HashSet::<&isize>::from_iter(x.iter()).len();
        let n_unique_y = std::collections::HashSet::<&isize>::from_iter(y.iter()).len();

        // Store number of skeleton points
        let n_skel = x.len();
        let mut indices = (0..n_skel).collect::<Vec<_>>();
        if n_unique_x > n_unique_y {
            indices.sort_by_key(|&i| &x[i]);
        } else {
            indices.sort_by_key(|&i| &y[i]);
        }

        let all_points =
            numpy::ndarray::Array2::from_shape_fn(
                (n_skel, 2),
                |(k, n)| {
                    if n == 0 {
                        x[k]
                    } else {
                        y[k]
                    }
                },
            );
        let mut remaining: Vec<_> = all_points.rows().into_iter().filter(|x| x != e1).collect();

        let mut points_sorted = numpy::ndarray::Array2::<isize>::zeros((n_skel, 2));
        points_sorted.row_mut(0).assign(&e1);
        points_sorted.row_mut(n_skel - 1).assign(&e2);
        for i in 1..n_skel {
            // Get the last sorted point from which we continue
            let p: numpy::ndarray::Array1<_> = points_sorted.row(i - 1).to_owned();
            // Check which remaining points do have distance == 1 to this point
            let mut total_diff = isize::MAX;
            let mut total_q = remaining[0];
            let mut total_index = 0;
            '_inner_loop: for (n, q) in remaining.iter().enumerate() {
                use core::ops::Sub;
                let diff = (q.sub(&p)).mapv(|x| x.abs()).sum();
                if diff == 1 {
                    total_q = *q;
                    total_index = n;
                    break '_inner_loop;
                } else if diff < total_diff {
                    total_diff = diff;
                    total_q = *q;
                    total_index = n;
                }
            }
            points_sorted.row_mut(i).assign(&total_q);
            remaining.remove(total_index);
        }
        Ok(points_sorted.to_pyarray(py))
    } else {
        Err(pyo3::exceptions::PyValueError::new_err(
            "Detected less than 2 endpoints after skeletonization",
        ))
    }
}

#[test]
fn test_generate_coordinates_1() {
    let p = ndarray::array![0.0, 0.0];
    let coords = generate_coordinates_sphere(&p.view(), 1.0, 0.0, core::f32::consts::PI, 1);
    let coord = coords[0];
    approx::assert_abs_diff_eq!(coord.x, 0.0);
    approx::assert_abs_diff_eq!(coord.y, 1.0);
    assert_eq!(coords.len(), 1);
}

#[test]
fn test_generate_coordinates_2() {
    let p = ndarray::array![0.0, 0.0];
    let coords = generate_coordinates_sphere(
        &p.view(),
        1.0,
        core::f32::consts::PI / 2.0,
        3.0 / 2.0 * core::f32::consts::PI,
        3,
    );
    let coord1 = coords[0];
    let coord2 = coords[1];
    let coord3 = coords[2];

    approx::assert_abs_diff_eq!(coord1.x, -1.0 / 2f32.sqrt(), epsilon = 0.001);
    approx::assert_abs_diff_eq!(coord1.y, 1.0 / 2f32.sqrt(), epsilon = 0.001);
    approx::assert_abs_diff_eq!(coord2.x, -1.0, epsilon = 0.001);
    approx::assert_abs_diff_eq!(coord2.y, 0.0, epsilon = 0.001);
    approx::assert_abs_diff_eq!(coord3.x, -1.0 / 2f32.sqrt(), epsilon = 0.001);
    approx::assert_abs_diff_eq!(coord3.y, -1.0 / 2f32.sqrt(), epsilon = 0.001);
    assert_eq!(coords.len(), 3);
}

#[test]
fn test_determine_spheroid_coordinates_between_rectangles_1() {
    //        p3
    //       /
    // p1---p2
    let p1 = ndarray::array![-1.0, 0.0];
    let p2 = ndarray::array![0.0, 0.0];
    let p3 = ndarray::array![
        1.0 / core::f32::consts::SQRT_2,
        1.0 / core::f32::consts::SQRT_2
    ];

    let (is_left, coords) = determine_spheroid_coordinates_between_rectangles(
        &p1.view(),
        &p2.view(),
        &p3.view(),
        0.5,
        8,
    );

    assert!(!is_left);
    assert_eq!(coords.len(), 1);
    approx::assert_abs_diff_eq!(coords[0].x, 0.5 * (core::f32::consts::PI / 8.0).sin());
    approx::assert_abs_diff_eq!(coords[0].y, -0.5 * (core::f32::consts::PI / 8.0).cos());
}

#[test]
fn test_determine_spheroid_coordinates_between_rectangles_2() {
    // p1
    // |
    // |
    // p2---p3
    let p1 = ndarray::array![0.0, 1.0];
    let p2 = ndarray::array![0.0, 0.0];
    let p3 = ndarray::array![1.0, 0.0];

    let (is_left, coords) = determine_spheroid_coordinates_between_rectangles(
        &p1.view(),
        &p2.view(),
        &p3.view(),
        0.9,
        16,
    );

    assert!(!is_left);
    assert_eq!(coords.len(), 2);
    let dangle = core::f32::consts::FRAC_PI_2 / 3.0;
    approx::assert_abs_diff_eq!(coords[0].x, -0.9 * dangle.cos());
    approx::assert_abs_diff_eq!(coords[0].y, -0.9 * dangle.sin());
    approx::assert_abs_diff_eq!(coords[1].x, -0.9 * (2.0 * dangle).cos());
    approx::assert_abs_diff_eq!(coords[1].y, -0.9 * (2.0 * dangle).sin());
}
