#![deny(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]
//! This crate solves a system containing bacterial rods in 2D.
//! The bacteria grow and divide thus resulting in a packed environment after short periods of
//! time.

mod agent;
mod datatypes;
mod fitting;
mod imaging;
mod sampling;
mod simulation;

pub use agent::*;
pub use cellular_raza::prelude::{CellIdentifier, VoxelPlainIndex};
use cellular_raza::prelude::{MiePotentialF32, MorsePotentialF32};
pub use datatypes::*;
pub use fitting::*;
pub use imaging::*;
pub use sampling::*;
pub use simulation::*;

use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "cr_mech_coli")]
fn cr_mech_coli(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_simulation_with_agents, m)?)?;
    m.add_function(wrap_pyfunction!(sort_cellular_identifiers, m)?)?;
    m.add_class::<CellIdentifier>()?;
    m.add_class::<VoxelPlainIndex>()?;

    m.add_function(wrap_pyfunction!(parents_diff_mask, m)?)?;
    m.add_function(wrap_pyfunction!(_sort_points, m)?)?;
    m.add_function(wrap_pyfunction!(counter_to_color, m)?)?;
    m.add_function(wrap_pyfunction!(color_to_counter, m)?)?;
    m.add_class::<Configuration>()?;
    m.add_class::<RodMechanicsSettings>()?;
    m.add_class::<MorsePotentialF32>()?;
    m.add_class::<MiePotentialF32>()?;
    m.add_class::<PhysicalInteraction>()?;
    m.add_class::<AgentSettings>()?;
    m.add_class::<RodAgent>()?;
    m.add_class::<CellContainer>()?;
    m.add_class::<CellIdentifier>()?;
    Ok(())
}
