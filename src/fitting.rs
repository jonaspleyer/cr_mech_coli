use crate::CellContainer;
use pyo3::prelude::*;

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

macro_rules! new_error (
    ($error_kind:ident, $($message:tt),*) => {
        pyo3::exceptions:: $error_kind ::new_err(format!($($message),*))
    };
);

/// Calculates the penalty in relative area difference accounting for a lower number when cells have
/// the same parent.
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
        .or_else(|e| Err(new_error!(PyValueError, "{e}")))?;
    let m2 = m2
        .to_shape(new_shape)
        .or_else(|e| Err(new_error!(PyValueError, "{e}")))?;
    let diff_mask = numpy::ndarray::Array1::<f32>::from_iter(
        m1.outer_iter()
            .zip(m2.outer_iter())
            .map(|(c1, c2)| {
                if c1 != c2 && c1.sum() != 0 && c2.sum() != 0 {
                    let c1 = [c1[0], c1[1], c1[2]];
                    let c2 = [c2[0], c2[1], c2[2]];

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
                    let p1 = cell_container.parent_map.get(&i1).ok_or(new_error!(
                        PyKeyError,
                        "could not find cell {:?}",
                        i1
                    ))?;
                    let p2 = cell_container.parent_map.get(&i2).ok_or(new_error!(
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
            .collect::<pyo3::PyResult<Vec<f32>>>()?
            .into_iter(),
    );
    let diff_mask = diff_mask
        .to_shape([s[0], s[1]])
        .or_else(|e| Err(pyo3::exceptions::PyValueError::new_err(format!("{e}"))))?;
    Ok(diff_mask.to_pyarray_bound(py))
}
