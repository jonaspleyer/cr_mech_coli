use pyo3::prelude::*;

/// A Python module implemented in Rust.
pub fn crm_estimate_params_rs(py: Python) -> PyResult<Bound<PyModule>> {
    let m = PyModule::new(
        py,
        "cr_mech_coli.crm_estimate_params.crm_estimate_params_rs",
    )?;
    Ok(m)
}
