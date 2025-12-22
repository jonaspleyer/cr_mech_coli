use std::collections::{btree_map::Entry, BTreeMap, HashMap};

use cellular_raza::prelude::CellIdentifier;
use itertools::Itertools;
use pyo3::prelude::*;

fn data_color_to_unique_ident(color: u8, data_iteration: usize) -> Option<u8> {
    // Black is background so no identifier should be provided
    if color == 0 {
        return None;
    }
    // Before iteration 8 there is no cell division
    if data_iteration <= 10 {
        Some(color)
    // After iteration 8 cell division has ocurred for all cells
    } else {
        match color {
            8 => Some(5),
            10 => Some(6),
            c => Some(c + 6),
        }
    }
}

fn unique_ident_to_parent_ident(unique_ident: u8) -> Option<u8> {
    match unique_ident {
        7 => Some(2),
        8 => Some(1),
        9 => Some(2),
        10 => Some(1),
        11 => Some(3),
        12 => Some(4),
        13 => Some(3),
        5 => None,
        6 => None,
        15 => Some(4),
        _ => None,
    }
}

fn match_parents(unique_ident: u8) -> PyResult<CellIdentifier> {
    if 0 < unique_ident && unique_ident < 7 {
        Ok(CellIdentifier::Initial(unique_ident as usize - 1))
    } else {
        Err(pyo3::exceptions::PyKeyError::new_err(format!(
            "Could not find parent ident for unique color {unique_ident}"
        )))
    }
}

#[pyfunction]
fn get_color_mappings(
    container: &crate::CellContainer,
    masks_data: Vec<numpy::PyReadonlyArray2<u8>>,
    iterations_data: Vec<usize>,
    positions_all: Vec<numpy::PyReadonlyArray3<f32>>,
) -> PyResult<(
    HashMap<u64, HashMap<u8, CellIdentifier>>,
    BTreeMap<(u8, u8, u8), CellIdentifier>,
    BTreeMap<CellIdentifier, Option<CellIdentifier>>,
)> {
    let daughter_map = container.get_daughter_map();
    let sim_iterations = container.get_all_iterations();
    let sim_iterations_subset = iterations_data
        .iter()
        .map(|i| sim_iterations[*i])
        .collect::<Vec<_>>();

    let mut parent_map = container.parent_map.clone();
    let mut color_to_cell = container.color_to_cell.clone();

    let mut all_mappings = HashMap::with_capacity(iterations_data.len());
    for (i, n) in iterations_data.iter().enumerate() {
        let sim_iter = sim_iterations[*n];
        let mask_data = &masks_data[i].as_array();

        let unique_colors: Vec<_> = mask_data
            .iter()
            .unique()
            .filter(|&x| *x != 0)
            .map(|c| data_color_to_unique_ident(*c, *n).map(|x| (*c, x)))
            .collect::<Result<_, _>>()?;

        // Cells which are daughters need to be mapped to the correct CellIdentifier
        // Cells which are not, can simply be mapped to the correct parent
        let mapping: HashMap<u8, CellIdentifier> = unique_colors
            .iter()
            .map(|(data_color, uid)| {
                if let Some(parent) = unique_ident_to_parent_ident(*uid) {
                    let p = positions_all[i]
                        .as_array()
                        .slice(ndarray::s![*data_color as usize - 1, .., ..])
                        .to_owned();

                    let parent_ident = match_parents(parent)?;
                    // If we do not find a parent, this may mean that the corresponding cell has
                    // not divided in the simulation yet.
                    if let Some(daughters_sim) = daughter_map.get(&parent_ident) {
                        if let Some(first_iter_sim) = daughters_sim
                            .iter()
                            .filter_map(|d| {
                                container
                                    .get_cell_history(*d)
                                    .0
                                    .into_keys()
                                    .filter(|k| sim_iterations_subset.contains(k))
                                    .min()
                            })
                            .max()
                        {
                            let n_daughter = daughters_sim
                                .iter()
                                .map(|d| {
                                    let pd = &container.get_cells_at_iteration(first_iter_sim)[d]
                                        .0
                                        .mechanics
                                        .pos;
                                    let mut dist1 = 0.0;
                                    let mut dist2 = 0.0;
                                    let ntotal = p.nrows();
                                    for i in 0..ntotal {
                                        dist1 += ((pd[(i, 0)] - p[(i, 0)]).powi(2)
                                            + (pd[(i, 1)] - p[(i, 1)]).powi(2))
                                        .sqrt();
                                        dist2 += ((pd[(i, 0)] - p[(ntotal - i - 1, 0)]).powi(2)
                                            + (pd[(i, 1)] - p[(ntotal - i - 1, 1)]).powi(2))
                                        .sqrt();
                                    }
                                    dist1.min(dist2)
                                })
                                .enumerate()
                                .min_by(|x, y| {
                                    x.1.partial_cmp(&y.1).unwrap_or(std::cmp::Ordering::Equal)
                                })
                                .map(|x| x.0);

                            if let Some(n) = n_daughter {
                                Ok((*data_color, daughters_sim[n]))
                            } else {
                                Err(pyo3::exceptions::PyValueError::new_err(
                                    format!("Daughter idents {daughters_sim:?} not present in simulation data."),
                                ))
                            }
                        } else {
                            Err(pyo3::exceptions::PyValueError::new_err(
                                format!("Parent ident {parent_ident:?} does not have daughters"),
                            ))
                        }
                    } else {
                        // Generate new key
                        let daughter_ident = CellIdentifier::new_inserted(
                            cellular_raza::prelude::VoxelPlainIndex(0),
                            parent_map.len() as u64,
                        );
                        parent_map.insert(daughter_ident, Some(parent_ident));

                        let mut counter = color_to_cell.len() as u32;

                        while (counter as usize) < color_to_cell.len() + 100 {
                            let new_color = crate::counter_to_color(counter);
                            if let Entry::Vacant(v) = color_to_cell.entry(new_color) {
                                v.insert(daughter_ident);
                                break;
                            } else {
                                counter += 1;
                            }
                        }
                        if (counter as usize) == color_to_cell.len() + 100 {
                            return Err(pyo3::exceptions::PyValueError::new_err(
                                "Loop for constructing new color exceeded 100 steps.",
                            ));
                        }

                        Ok((*data_color, daughter_ident))
                    }
                } else {
                    Ok((*data_color, match_parents(*uid)?))
                }
            })
            .collect::<Result<_, _>>()?;

        all_mappings.insert(sim_iter, mapping);

        // Now all images have colors which always match to the same cell
        // println!(
        //     "cells sim: {} cells data: {} {unique_colors:?} {:?}",
        //     container.get_cells_at_iteration(sim_iter).len(),
        //     unique_colors.len(),
        //     cellidents,
        // );
    }

    Ok((all_mappings, color_to_cell, parent_map))
}

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "crm_divide_rs", module = "cr_mech_coli.crm_divide_rs")]
pub fn crm_divide_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_color_mappings, m)?)?;
    Ok(())
}
