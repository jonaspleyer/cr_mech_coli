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
            1 => Some(7),
            2 => Some(8),
            3 => Some(9),
            4 => Some(10),
            5 => Some(11),
            6 => Some(12),
            7 => Some(13),
            9 => Some(14),
            _ => None,
        }
    }
}

fn unique_ident_to_parent_ident(unique_ident: u8) -> Option<u8> {
    match unique_ident {
        5 => None,
        6 => None,
        7 => Some(2),
        8 => Some(1),
        9 => Some(2),
        10 => Some(1),
        11 => Some(3),
        12 => Some(4),
        13 => Some(3),
        14 => Some(4),
        _ => None,
    }
}

fn unique_ident_get_daughters(unique_ident: u8) -> Option<(u8, u8)> {
    match unique_ident {
        1 => Some((8, 10)),
        2 => Some((7, 9)),
        3 => Some((11, 13)),
        4 => Some((12, 14)),
        5 => None,
        6 => None,
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

fn determine_from_container(
    container: &crate::CellContainer,
    daughters_sim: &[CellIdentifier],
    p: &ndarray::Array2<f32>,
    sim_iterations_subset: &[u64],
) -> PyResult<CellIdentifier> {
    // ==== Determine which daughter fits best ================================
    let first_iter_sim = daughters_sim
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
        // Almost impossible to return nothing. But we simply make sure. We do
        // not want to throw any edge cases during optimization and crash
        // everything.
        .ok_or(pyo3::exceptions::PyValueError::new_err(
            "Daughters not present in simulation despite listing",
        ))?;

    // ==== Determine which daughter fits best ================================
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
                dist1 +=
                    ((pd[(i, 0)] - p[(i, 0)]).powi(2) + (pd[(i, 1)] - p[(i, 1)]).powi(2)).sqrt();
                dist2 += ((pd[(i, 0)] - p[(ntotal - i - 1, 0)]).powi(2)
                    + (pd[(i, 1)] - p[(ntotal - i - 1, 1)]).powi(2))
                .sqrt();
            }
            dist1.min(dist2)
        })
        .enumerate()
        .min_by(|x, y| x.1.partial_cmp(&y.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|x| x.0);

    // ==== If the previous comparison does not return, we return an error ====
    // ==== This is very unlikely however ... =================================
    let n = n_daughter.ok_or(pyo3::exceptions::PyValueError::new_err(format!(
        "Daughter idents {daughters_sim:?} not present in simulation data."
    )))?;
    Ok(daughters_sim[n])
}

/// Cells which are daughters need to be mapped to the correct CellIdentifier
/// Cells which are not, can simply be mapped to the correct parent
///
/// 1. Transform data_color to unique identifier
/// 2. If unique ident has no parent, cell must be initial
///    => map unique ident directly to parent
/// 3. Else If parent has daughters in simulation
///    => determine which daughter is most likely to be the cell itself (compare positions)
///    => create mapping from unique ident to the chosen daughter
/// 4. Else If parent exists but has no daughters in simulation
///    => We check if we have already inserted artificial new idents and use them
///    => Otherwise we begin to artificially insert new idents
///      i)   Determine unique IDs of daughters
///      ii)  Create two new artificial CellIdentifiers (inserted) and insert them
///      iii) Link the lower/higher unique id with the lower/higher artificial
///           CellIdentifier
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
            .filter_map(|c| data_color_to_unique_ident(*c, *n).map(|x| (*c, x)))
            .collect::<_>();

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
                        let d = determine_from_container(
                            container,
                            daughters_sim,
                            &p,
                            &sim_iterations_subset,
                        )?;
                        Ok((*data_color, d))
                    } else {
                        // No daughter is present in the simulation. We nevertheless check if the
                        // parent has some daughters in our new parent map.
                        // If this is the case, we can reuse these colors. Furthermore, we need to
                        // check that we have not already used the same daughter color mapping in
                        // this iteration before.

                        // Check if new CellIdents are already there
                        let existing: Vec<_> = parent_map
                            .iter()
                            .filter(|(_, v)| v == &&Some(parent_ident))
                            .collect();

                        let (d1, d2) = if !existing.is_empty() {
                            let id1: CellIdentifier = *existing[0].0;
                            let id2: CellIdentifier = *existing[1].0;
                            if id1 < id2 {
                                PyResult::Ok((id1, id2))
                            } else {
                                PyResult::Ok((id2, id1))
                            }
                        } else {
                            let mut create_and_insert_ident = || {
                                let daughter_ident = CellIdentifier::new_inserted(
                                    cellular_raza::prelude::VoxelPlainIndex(0),
                                    // This small offset ensures that we construct a new index
                                    parent_map.len() as u64 + 1,
                                );
                                parent_map.insert(daughter_ident, Some(parent_ident));

                                // Generate new color.
                                // This loop is to ensure that no new color is chosen by accident.
                                // We limit the loop to 100 iterations. This should be well more
                                // than enough to find a new color.
                                // If it does not finish, we return an error.
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

                                if (counter as usize) == color_to_cell.len() + 99 {
                                    return Err(pyo3::exceptions::PyValueError::new_err(
                                        "Loop for constructing new color exceeded 100 steps.",
                                    ));
                                }

                                Ok(daughter_ident)
                            };
                            let id1: CellIdentifier = create_and_insert_ident()?;
                            let id2: CellIdentifier = create_and_insert_ident()?;
                            Ok((id1, id2))
                        }?;

                        // We now have two possible candiates to pick from.
                        // Both are not in the simulation so far.
                        // Therefore we obtain the other uid of the sister cell and determine which
                        // one is higher.
                        // Since CellIdentifier also implements PartialEq we can use this to
                        // compare them.

                        let uid_daughters = unique_ident_get_daughters(parent).ok_or(
                            pyo3::exceptions::PyKeyError::new_err(format!(
                                "Can not find daughters to parent {parent}"
                            )),
                        )?;
                        let daughter_ident = if *uid == uid_daughters.0 {
                            d1
                        } else if *uid == uid_daughters.1 {
                            d2
                        } else {
                            return Err(pyo3::exceptions::PyValueError::new_err(
                                "Loop for constructing new color exceeded 100 steps.",
                            ));
                        };

                        Ok((*data_color, daughter_ident))
                    }
                } else {
                    Ok((*data_color, match_parents(*uid)?))
                }
            })
            .collect::<Result<_, _>>()?;

        all_mappings.insert(sim_iter, mapping);
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

#[test]
fn test_unique_idents_for_colors() {
    for iteration in 0..30 {
        let mut previous_ids = Vec::<u8>::new();
        for data_color in 0..10 {
            let uid = data_color_to_unique_ident(data_color, iteration);
            if data_color == 0 {
                assert!(uid.is_none());
            } else if uid.is_none() {
                assert_eq!(data_color, 0);
            } else {
                assert!(uid.is_some());
                let u = uid.unwrap();
                assert!(u <= 14);
                previous_ids.iter().all(|x| *x < u);
                previous_ids.push(u);
            }
        }
    }
}

#[test]
fn test_parent_idents() {
    for unique_ident in 1..15 {
        let parent = unique_ident_to_parent_ident(unique_ident);
        if unique_ident > 6 {
            assert!(parent.is_some());
        }
    }
}
