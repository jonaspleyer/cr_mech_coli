use cellular_raza::prelude::CellIdentifier;
use pyo3::prelude::*;
use std::collections::HashMap;

/// Manages all information resulting from an executed simulation
#[pyclass]
pub struct CellContainer {
    /// Contains snapshots of all cells at each saved step
    #[pyo3(get)]
    pub cells: HashMap<u64, HashMap<CellIdentifier, (PyObject, Option<CellIdentifier>)>>,
    /// Maps each cell to its parent if existent
    #[pyo3(get)]
    pub parent_map: HashMap<CellIdentifier, Option<CellIdentifier>>,
    /// Maps each cell to its children
    #[pyo3(get)]
    pub child_map: HashMap<CellIdentifier, Vec<CellIdentifier>>,
    /// Maps each cell to its color
    #[pyo3(get)]
    pub cell_to_color: HashMap<CellIdentifier, [u8; 3]>,
    /// Maps each color back to its cell
    #[pyo3(get)]
    pub color_to_cell: HashMap<[u8; 3], CellIdentifier>,
}

#[pymethods]
impl CellContainer {
    /// Constructs a new :class:`CellContainer` from the history of objects.
    #[new]
    pub fn new(
        all_cells: HashMap<u64, HashMap<CellIdentifier, (PyObject, Option<CellIdentifier>)>>,
    ) -> pyo3::PyResult<Self> {
        let cells = all_cells;
        let cell_container = Self {
            cells,
            parent_map: HashMap::new(),
            child_map: HashMap::new(),
            cell_to_color: HashMap::new(),
            color_to_cell: HashMap::new(),
        };
        let all_cells = cell_container.get_cells();
        let parent_map: HashMap<CellIdentifier, Option<CellIdentifier>> = all_cells
            .into_iter()
            .flat_map(|(_, cells)| cells.into_iter())
            .map(|(ident, (_, parent))| (ident, parent))
            .collect();
        let mut identifiers: Vec<_> = parent_map.iter().map(|(i, _)| i.clone()).collect();
        identifiers.sort();
        let cell_to_color: HashMap<_, _> = Self::assign_colors_to_cells(identifiers)?;
        let color_to_cell: HashMap<_, _> = cell_to_color
            .clone()
            .into_iter()
            .map(|(x, y)| (y, x))
            .collect();
        let child_map = parent_map
            .iter()
            .filter_map(|(child, parent)| parent.map(|x| (x, child)))
            .fold(HashMap::new(), |mut acc, (parent, &child)| {
                acc.entry(parent).or_insert(vec![child]).push(child);
                acc
            });
        Ok(Self {
            parent_map,
            child_map,
            cell_to_color,
            color_to_cell,
            ..cell_container
        })
    }

    /// Get all cells at all iterations
    ///
    /// Returns:
    ///     dict[int, dict[CellIdentifier, tuple[PyObject, CellIdentifier | None]]: A dictionary
    ///     containing all cells with their identifiers, values and possible parent identifiers for
    ///     every iteration.
    pub fn get_cells(
        &self,
    ) -> HashMap<u64, HashMap<CellIdentifier, (PyObject, Option<CellIdentifier>)>> {
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
    ) -> HashMap<CellIdentifier, (PyObject, Option<CellIdentifier>)> {
        self.cells
            .get(&iteration)
            .cloned()
            .or(Some(HashMap::new()))
            .unwrap()
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
    ) -> (HashMap<u64, PyObject>, Option<CellIdentifier>) {
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
    pub fn get_color(&self, identifier: &CellIdentifier) -> Option<[u8; 3]> {
        self.cell_to_color.get(identifier).copied()
    }

    /// Obtains the cell which had been assigned this color
    ///
    /// Args:
    ///     color(tuple[int, int, int]): A tuple (or list) with 3 8bit values
    /// Returns:
    ///     CellIdentifier | None: The identifier of the cell
    pub fn get_cell_from_color(&self, color: [u8; 3]) -> Option<CellIdentifier> {
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
    pub fn get_parent_map(&self) -> HashMap<CellIdentifier, Option<CellIdentifier>> {
        self.parent_map.clone()
    }

    /// A dictionary mapping each cell to its children
    pub fn get_child_map(&self) -> HashMap<CellIdentifier, Vec<CellIdentifier>> {
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
        self.parent_map
            .iter()
            .map(|(ident, _)| ident.clone())
            .collect()
    }

    /// This functions assigns unique colors to given cellular identifiers.
    /// Used in :mod:`cr_mech_coli.imaging` techniques.
    ///
    /// Args:
    ///     cell_container (dict): An instance of the :class:`CellContainer` class.
    /// Returns:
    ///     dict: A dictionary mapping :class:`CellIdentifier` to colors.
    #[staticmethod]
    pub fn assign_colors_to_cells(
        identifiers: Vec<CellIdentifier>,
    ) -> PyResult<HashMap<CellIdentifier, [u8; 3]>> {
        let mut color_counter = 1;
        let mut colors = HashMap::new();
        let mut err = Ok(());
        for ident in identifiers.iter() {
            colors.entry(ident.clone()).or_insert_with(|| {
                let color = crate::imaging::counter_to_color(color_counter);
                color_counter += 1;
                if color_counter > 251u32.pow(3) {
                    err = Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Too many identifiers: {} MAX: {}.",
                        identifiers.len(),
                        251u32.pow(3)
                    )));
                }
                color
            });
        }
        match err {
            Ok(()) => Ok(colors),
            Err(e) => Err(e),
        }
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
            .clone())
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
}