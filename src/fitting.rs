use crate::CellContainer;
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
///     cell_container(CellContainer): See :class:`CellContainer`
///     parent_penalty(float): Penalty value when one cell is daughter of other.
///         Should be between 0 and 1.
#[pyfunction]
#[pyo3(signature = (mask1, mask2, cell_container, parent_penalty = 0.5))]
pub fn parents_diff_mask<'py>(
    py: Python<'py>,
    mask1: numpy::PyReadonlyArray3<'py, u8>,
    mask2: numpy::PyReadonlyArray3<'py, u8>,
    cell_container: &CellContainer,
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

                    let i1 = cell_container.color_to_cell.get(&c1).ok_or(new_error!(
                        PyKeyError,
                        "could not find color {:?}",
                        c1
                    ))?;
                    let i2 = cell_container.color_to_cell.get(&c2).ok_or(new_error!(
                        PyKeyError,
                        "could not find color {:?}",
                        c2
                    ))?;

                    // Check if one is the parent of the other
                    let p1 = cell_container.parent_map.get(i1).ok_or(new_error!(
                        PyKeyError,
                        "could not find cell {:?}",
                        i1
                    ))?;
                    let p2 = cell_container.parent_map.get(i2).ok_or(new_error!(
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
            '_inner_loop: for n in 0..remaining.len() {
                let q = remaining[n];
                use core::ops::Sub;
                let diff = (q.sub(&p)).mapv(|x| x.abs()).sum();
                if diff == 1 {
                    total_q = q;
                    total_index = n;
                    // points_sorted.row_mut(i).assign(&q);
                    // remaining.remove(n);
                    break '_inner_loop;
                } else if diff < total_diff {
                    total_diff = diff;
                    total_q = q;
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
