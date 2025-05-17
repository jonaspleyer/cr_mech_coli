use pyo3::prelude::*;

mod optimize;
mod predict;
mod settings;

use optimize::*;
pub use settings::*;

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "crm_fit_rs", module = "cr_mech_coli.crm_fit")]
pub fn crm_fit_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SampledFloat>()?;
    m.add_class::<Parameter>()?;
    m.add_class::<Constants>()?;
    m.add_class::<Parameters>()?;
    m.add_class::<Settings>()?;
    m.add_class::<Others>()?;
    m.add_class::<PotentialType>()?;
    m.add_class::<OptimizationResult>()?;
    m.add_class::<OptimizationMethod>()?;
    m.add_class::<DifferentialEvolution>()?;
    m.add_class::<OptimizationInfos>()?;
    m.add_function(wrap_pyfunction!(predict::run_simulation, m)?)?;
    m.add_function(wrap_pyfunction!(predict::predict_calculate_cost, m)?)?;
    m.add_function(wrap_pyfunction!(optimize::run_optimizer, m)?)?;
    Ok(())
}
