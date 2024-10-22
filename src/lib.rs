#![deny(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]
//! This crate solves a system containing bacterial rods in 2D.
//! The bacteria grow and divide thus resulting in a packed environment after short periods of
//! time.

mod imaging;
mod sampling;
mod simulation;

pub use cellular_raza::prelude::{CellIdentifier, VoxelPlainIndex};
pub use imaging::*;
pub use sampling::*;
pub use simulation::*;

use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
fn cr_mech_coli_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_simulation, m)?)?;
    m.add_function(wrap_pyfunction!(sort_cellular_identifiers, m)?)?;
    m.add_class::<CellIdentifier>()?;
    m.add_class::<VoxelPlainIndex>()?;

    m.add_function(wrap_pyfunction!(counter_to_color, m)?)?;
    m.add_function(wrap_pyfunction!(color_to_counter, m)?)?;
    m.add_class::<Configuration>()?;
    m.add_class::<RodMechanicsSettings>()?;
    m.add_class::<AgentSettings>()?;
    m.add_class::<RodAgent>()?;
    m.add_class::<SimResult>()?;
    Ok(())
}
