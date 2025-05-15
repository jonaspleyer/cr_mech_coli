use pyo3::prelude::*;

/// A Python module implemented in Rust.
pub fn crm_estimate_params_rs(py: Python) -> PyResult<Bound<PyModule>> {
    let m = PyModule::new(py, "crm_estimate_params")?;
    Ok(m)
}
