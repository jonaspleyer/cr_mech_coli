#![deny(missing_docs)]
//! This crate solves a system containing bacterial rods in 2D.
//! The bacteria grow and divide thus resulting in a packed environment after short periods of
//! time.

mod agent;
mod cell_container;
mod config;
mod fitting;
mod imaging;
#[cfg(feature = "vtk")]
mod imaging_vtk;

/// Fit data to publication by (Amir et. al. 2014)
pub mod crm_amir;
/// Script for performing parameter estimations with the image-space matric
pub mod crm_divide;
/// Estimate parameters with classical methods from data
pub mod crm_estimate_params;
/// Fit a mechanical model to microscopic images
pub mod crm_fit;
/// Predict Multilayer Behaviour
pub mod crm_multilayer;
/// Functions to execute simulations
pub mod simulation;

pub use agent::*;
pub use cell_container::*;
pub use cellular_raza::prelude::{CellIdentifier, VoxelPlainIndex};
use cellular_raza::prelude::{MiePotentialF32, MorsePotentialF32, StorageOption};
pub use config::*;
pub use fitting::*;
pub use imaging::*;
#[cfg(feature = "vtk")]
pub use imaging_vtk::*;
pub use simulation::*;

use pyo3::prelude::*;

/// Micro metre base unit
pub const MICRO_METRE: f32 = 1.0;
/// Minute in base unit
pub const MINUTE: f32 = 1.0;
/// Hour derived from [MINUTE]
pub const HOUR: f32 = 60. * MINUTE;

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "cr_mech_coli")]
fn cr_mech_coli(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule = PyModule::new(py, "cr_mech_coli.crm_fit.crm_fit_rs")?;
    crm_divide::crm_divide_rs(&submodule)?;
    m.add_submodule(&submodule)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("cr_mech_coli.crm_divide.crm_divide_rs", submodule)?;

    let submodule = PyModule::new(py, "cr_mech_coli.crm_fit.crm_fit_rs")?;
    crm_fit::crm_fit_rs(&submodule)?;
    m.add_submodule(&submodule)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("cr_mech_coli.crm_fit.crm_fit_rs", submodule)?;

    let submodule_multilayer = crm_multilayer::crm_multilayer_rs(py)?;
    m.add_submodule(&submodule_multilayer)?;
    py.import("sys")?.getattr("modules")?.set_item(
        "cr_mech_coli.crm_multilayer.crm_multilayer_rs",
        submodule_multilayer,
    )?;

    let submodule_fit_amir = crm_amir::crm_amir(py)?;
    m.add_submodule(&submodule_fit_amir)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("cr_mech_coli.crm_amir.crm_amir_rs", submodule_fit_amir)?;

    let submodule_estimate_params = crm_estimate_params::crm_estimate_params_rs(py)?;
    m.add_submodule(&submodule_estimate_params)?;
    py.import("sys")?.getattr("modules")?.set_item(
        "cr_mech_coli.crm_estimate_params.crm_estimate_params_rs",
        submodule_estimate_params,
    )?;

    m.add_function(wrap_pyfunction!(generate_positions, m)?)?;
    m.add_function(wrap_pyfunction!(generate_agents, m)?)?;
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
    m.add_class::<GrowthRateSetter>()?;
    m.add_class::<SpringLengthThresholdSetter>()?;
    m.add_class::<RodAgent>()?;
    m.add_class::<CellContainer>()?;
    m.add_class::<CellIdentifier>()?;
    m.add_class::<StorageOption>()?;

    m.add("MINUTE", MINUTE)?;
    m.add("HOUR", HOUR)?;
    m.add("MICRO_METRE", MICRO_METRE)?;

    #[cfg(feature = "vtk")]
    m.add_function(wrap_pyfunction!(render_mask_rs, m)?)?;
    Ok(())
}
