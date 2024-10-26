use crate::{color_to_counter, CellContainer};
use cellular_raza::prelude::CellIdentifier;
use pyo3::prelude::*;
use std::collections::HashMap;

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

/// Calculates the penalty in relative area difference accounting for a lower number when cells have
/// the same parent.
#[pyfunction]
pub fn penalty_area_diff_account_parents<'py>(
    mask1: numpy::PyReadonlyArray3<'py, u8>,
    mask2: numpy::PyReadonlyArray3<'py, u8>,
    cell_container: &CellContainer,
    colors: HashMap<CellIdentifier, [u8; 3]>,
    penalty_with_parent: f32,
) -> pyo3::PyResult<f32> {
    use numpy::*;
    let m1 = mask1.as_array();
    let m2 = mask2.as_array();
    check_shape_identical_nonempty!(m1, m2);
    let new_shape = [m1.shape()[0] * m1.shape()[1], m1.shape()[2]];
    let m1 = m1
        .to_shape(new_shape)
        .or_else(|e| Err(pyo3::exceptions::PyValueError::new_err(format!("{e}"))))?;
    let m2 = m2
        .to_shape(new_shape)
        .or_else(|e| Err(pyo3::exceptions::PyValueError::new_err(format!("{e}"))))?;
    let mut color_to_parent_color = HashMap::<[u8; 3], [u8; 3]>::new();
    let penalty: f32 = m1
        .axis_iter(numpy::ndarray::Axis(0))
        .zip(m2.axis_iter(numpy::ndarray::Axis(0)))
        .map(|(c1, c2)| -> pyo3::PyResult<f32> {
            let c1 = [c1[0], c1[1], c1[2]];
            let c2 = [c2[0], c2[1], c2[2]];
            // First check if this color combination has already been calculated
            let mut get_parent = |color| -> pyo3::PyResult<Option<[u8; 3]>> {
                match color_to_parent_color.get(&color) {
                    Some(p) => Ok(Some(p.clone())),
                    None => {
                        let counter = color_to_counter(c1);
                        let ident = cell_container.counter_to_cell_identifier(counter)?;
                        let parent_ident = cell_container.get_parent(&ident)?;
                        match parent_ident {
                            Some(p) => {
                                let color_parent = colors.get(&p);
                                match color_parent {
                                    Some(cp) => {
                                        color_to_parent_color.insert(color, cp.clone());
                                    }
                                    None => (),
                                }
                                Ok(color_parent.copied())
                            }
                            None => Ok(None),
                        }
                    }
                }
            };
            let p1 = get_parent(c1)?;
            let p2 = get_parent(c2)?;
            if p1 == p2 {
                println!("[x] got parent");
                Ok(penalty_with_parent)
            } else {
                println!("[ ] no parent");
                Ok(1.0)
            }
        })
        .collect::<Result<Vec<f32>, _>>()?
        .into_iter()
        .sum();

    let s = m1.shape();
    let size = s[0] as u64 * s[1] as u64;
    Ok(penalty as f32 / size as f32)
}
