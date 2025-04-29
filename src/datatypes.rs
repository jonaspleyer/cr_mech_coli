use cellular_raza::prelude::{CellBox, CellIdentifier, StorageInterfaceLoad};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::{counter_to_color, Configuration, RodAgent};

/// Manages all information resulting from an executed simulation
#[pyclass]
#[derive(Clone, Deserialize, Serialize)]
pub struct CellContainer {
    /// Contains snapshots of all cells at each saved step
    #[pyo3(get)]
    pub cells: BTreeMap<u64, BTreeMap<CellIdentifier, (crate::RodAgent, Option<CellIdentifier>)>>,
    /// Maps each cell to its parent if existent
    #[pyo3(get)]
    pub parent_map: BTreeMap<CellIdentifier, Option<CellIdentifier>>,
    /// Maps each cell to its children
    #[pyo3(get)]
    pub child_map: BTreeMap<CellIdentifier, Vec<CellIdentifier>>,
    /// Maps each cell to its color
    #[pyo3(get)]
    pub cell_to_color: BTreeMap<CellIdentifier, (u8, u8, u8)>,
    /// Maps each color back to its cell
    #[pyo3(get)]
    pub color_to_cell: BTreeMap<(u8, u8, u8), CellIdentifier>,
}

#[pymethods]
impl CellContainer {
    /// Constructs a new :class:`CellContainer` from the history of objects.
    #[new]
    pub fn new(
        all_cells: BTreeMap<
            u64,
            BTreeMap<CellIdentifier, (crate::RodAgent, Option<CellIdentifier>)>,
        >,
    ) -> pyo3::PyResult<Self> {
        let cells = all_cells;
        let parent_map: BTreeMap<CellIdentifier, Option<CellIdentifier>> = cells
            .clone()
            .into_iter()
            .flat_map(|(_, cells)| cells.into_iter())
            .map(|(ident, (_, parent))| (ident, parent))
            .collect();
        let mut identifiers: Vec<_> = parent_map.clone().into_keys().collect();
        identifiers.sort();
        let cell_to_color: BTreeMap<_, _> = identifiers
            .into_iter()
            .enumerate()
            .map(|(n, ident)| (ident, counter_to_color(n as u32 + 1)))
            .collect();
        let color_to_cell: BTreeMap<_, _> = cell_to_color
            .clone()
            .into_iter()
            .map(|(x, y)| (y, x))
            .collect();
        let child_map = parent_map
            .iter()
            .filter_map(|(child, parent)| parent.map(|x| (x, child)))
            .fold(BTreeMap::new(), |mut acc, (parent, &child)| {
                acc.entry(parent).or_insert(vec![child]).push(child);
                acc
            });
        Ok(Self {
            cells,
            parent_map,
            child_map,
            cell_to_color,
            color_to_cell,
        })
    }

    /// Returns an identical clone
    pub fn __deepcopy__(&self, _memo: pyo3::Bound<pyo3::types::PyDict>) -> Self {
        self.clone()
    }

    /// Get all cells at all iterations
    ///
    /// Returns:
    ///     dict[int, dict[CellIdentifier, tuple[PyObject, CellIdentifier | None]]: A dictionary
    ///     containing all cells with their identifiers, values and possible parent identifiers for
    ///     every iteration.
    pub fn get_cells(
        &self,
    ) -> BTreeMap<u64, BTreeMap<CellIdentifier, (crate::RodAgent, Option<CellIdentifier>)>> {
        self.cells.clone()
    }

    /// Get cells at a specific iteration.
    ///
    /// Args:
    ///     iteration (int): Positive integer of simulation step iteration.
    /// Returns:
    ///     cells (dict): A dictionary mapping identifiers to the cell and its possible parent.
    /// Raises:
    ///     SimulationError: Generic error related to `cellular_raza <https://cellular-raza.com>`_
    ///     if any of the internal methods returns an error.
    pub fn get_cells_at_iteration(
        &self,
        iteration: u64,
    ) -> BTreeMap<CellIdentifier, (crate::RodAgent, Option<CellIdentifier>)> {
        self.cells
            .get(&iteration)
            .cloned()
            .unwrap_or(BTreeMap::new())
    }

    /// Load the history of a single cell
    ///
    /// Args:
    ///     identifier(CellIdentifier): The identifier of the cell in question
    /// Returns:
    ///     tuple[dict[int, PyObject], CellIdentifier | None]: A dictionary with all timespoints
    ///     and the cells confiruation at this time-point. Also returns the parent
    ///     :class:`CellIdentifier` if present.
    pub fn get_cell_history(
        &self,
        identifier: CellIdentifier,
    ) -> (BTreeMap<u64, crate::RodAgent>, Option<CellIdentifier>) {
        let mut parent = None;
        let hist = self
            .cells
            .clone()
            .into_iter()
            .filter_map(|(iteration, mut cells)| {
                cells.remove(&identifier).map(|(x, p)| {
                    parent = p;
                    (iteration, x)
                })
            })
            .collect();
        (hist, parent)
    }

    /// Obtain all iterations as a sorted list.
    pub fn get_all_iterations(&self) -> Vec<u64> {
        let mut iterations: Vec<_> = self.cells.iter().map(|(&it, _)| it).collect();
        iterations.sort();
        iterations
    }

    /// Obtains the parent identifier of a cell if it had a parent.
    ///
    /// Args:
    ///     identifier(CellIdentifier): The cells unique identifier
    /// Returns:
    ///     CellIdentifier | None: The parents identifier or :class:`None`
    pub fn get_parent(&self, identifier: &CellIdentifier) -> PyResult<Option<CellIdentifier>> {
        // Check the first iteration
        Ok(self
            .parent_map
            .get(identifier)
            .ok_or(pyo3::exceptions::PyKeyError::new_err(format!(
                "No CellIdentifier {:?} in map",
                identifier
            )))?
            .clone())
    }

    /// Obtains all children of a given cell
    ///
    /// Args:
    ///     identifier(CellIdentifier): The cells unique identifier
    /// Returns:
    ///     list[CellIdentifier]: All children of the given cell
    pub fn get_children(&self, identifier: &CellIdentifier) -> PyResult<Vec<CellIdentifier>> {
        Ok(self
            .child_map
            .get(identifier)
            .ok_or(pyo3::exceptions::PyKeyError::new_err(format!(
                "No CellIdentifier {:?} in map",
                identifier
            )))?
            .clone())
    }

    /// Obtains the color assigned to the cell
    ///
    /// Args:
    ///     identifier(CellIdentifier): The cells unique identifier
    /// Returns:
    ///     tuple[int, int, int] | None: The assigned color
    pub fn get_color(&self, identifier: &CellIdentifier) -> Option<(u8, u8, u8)> {
        self.cell_to_color.get(identifier).copied()
    }

    /// Obtains the cell which had been assigned this color
    ///
    /// Args:
    ///     color(tuple[int, int, int]): A tuple (or list) with 3 8bit values
    /// Returns:
    ///     CellIdentifier | None: The identifier of the cell
    pub fn get_cell_from_color(&self, color: (u8, u8, u8)) -> Option<CellIdentifier> {
        self.color_to_cell.get(&color).copied()
    }

    /// Determines if two cells share a common parent
    pub fn cells_are_siblings(&self, ident1: &CellIdentifier, ident2: &CellIdentifier) -> bool {
        let px1 = self.parent_map.get(ident1);
        let px2 = self.parent_map.get(ident2);
        if let (Some(p1), Some(p2)) = (px1, px2) {
            p1 == p2
        } else {
            false
        }
    }

    /// A dictionary mapping each cell to its parent
    pub fn get_parent_map(&self) -> BTreeMap<CellIdentifier, Option<CellIdentifier>> {
        self.parent_map.clone()
    }

    /// A dictionary mapping each cell to its children
    pub fn get_child_map(&self) -> BTreeMap<CellIdentifier, Vec<CellIdentifier>> {
        self.child_map.clone()
    }

    /// Returns all :class:`CellIdentifier` used in the simulation sorted in order.
    pub fn get_all_identifiers(&self) -> Vec<CellIdentifier> {
        let mut idents = self.get_all_identifiers_unsorted();
        idents.sort();
        idents
    }

    /// Identical to :func:`CellContainer.get_all_identifiers` but returns unsorted list.
    pub fn get_all_identifiers_unsorted(&self) -> Vec<CellIdentifier> {
        self.parent_map.clone().into_keys().collect()
    }

    /// Obtains the cell corresponding to the given counter of this simulation
    /// Used in :mod:`cr_mech_coli.imaging` techniques.
    ///
    /// Args:
    ///     counter(int): Counter of some cell
    /// Returns:
    ///     CellIdentifier: The unique identifier associated with this counter
    pub fn counter_to_cell_identifier(&self, counter: u32) -> pyo3::PyResult<CellIdentifier> {
        let identifiers = self.get_all_identifiers();
        Ok(identifiers
            .get(counter as usize)
            .ok_or(pyo3::exceptions::PyKeyError::new_err(format!(
                "Cannot assign CellIdentifier to counter {}",
                counter
            )))?
            .copy())
    }

    /// Get the :class:`CellIdentifier` associated to the given counter.
    /// Used in :mod:`cr_mech_coli.imaging` techniques.
    ///
    pub fn cell_identifier_to_counter(&self, identifier: &CellIdentifier) -> pyo3::PyResult<u32> {
        let identifiers = self.get_all_identifiers();
        for (i, ident) in identifiers.iter().enumerate() {
            if identifier == ident {
                return Ok(i as u32);
            }
        }
        Err(pyo3::exceptions::PyKeyError::new_err(format!(
            "No CellIdentifier {:?} in map",
            identifier
        )))
    }

    /// Serializes the :class:`CellContainer` into json format.
    pub fn serialize(&self) -> pyo3::PyResult<Vec<u8>> {
        let res: Vec<u8> = serde_pickle::to_vec(&self.cells, Default::default())
            // let res: String = serde_json::to_string(&self)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
        Ok(res)
    }

    /// Deserializes the :class`CellContainer` from a json string.
    #[staticmethod]
    pub fn deserialize(value: Vec<u8>) -> pyo3::PyResult<Self> {
        let cells = serde_pickle::from_slice(&value, Default::default())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("serde({e})")))?;
        CellContainer::new(cells)
    }

    /// Loads a saved result as a :class:`CellContainer`.
    #[staticmethod]
    pub fn load_from_storage(
        config: Configuration,
        date: std::path::PathBuf,
    ) -> Result<Self, cellular_raza::prelude::SimulationError> {
        let mut new_config = config.clone();
        new_config.storage_options = config
            .storage_options
            .clone()
            .into_iter()
            .filter(|x| x != &cellular_raza::prelude::StorageOption::Memory)
            .collect();
        let builder = crate::simulation::new_storage_builder(&new_config)
            .suffix(std::path::PathBuf::from("cells"))
            .init_with_date(&date);
        let cells_storage = cellular_raza::prelude::StorageManager::<
            CellIdentifier,
            (CellBox<RodAgent>, serde::de::IgnoredAny),
        >::open_or_create(builder, 0)
        .unwrap();
        let cells = cells_storage
            .load_all_elements()?
            .into_iter()
            .map(|(iteration, cells)| {
                (
                    iteration,
                    cells
                        .into_iter()
                        .map(|(ident, (cbox, _))| (ident, (cbox.cell, cbox.parent)))
                        .collect(),
                )
            })
            .collect();
        Ok(CellContainer::new(cells).unwrap())
    }
}
