use std::ops::Deref;

use super::settings::*;

use numpy::{PyUntypedArrayMethods, ToPyArray};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

#[pyclass(get_all, set_all)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub params: Vec<f32>,
    pub cost: f32,
    pub success: Option<bool>,
    pub neval: Option<usize>,
    pub niter: Option<usize>,
}

#[pymethods]
impl OptimizationResult {
    fn save_to_file(&self, filename: std::path::PathBuf) -> PyResult<()> {
        use std::io::prelude::*;
        let output = toml::to_string_pretty(&self)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
        let mut file = std::fs::File::create_new(filename)?;
        file.write_all(output.as_bytes())?;
        Ok(())
    }

    #[staticmethod]
    fn load_from_file(filename: std::path::PathBuf) -> PyResult<Self> {
        let contents = std::fs::read_to_string(filename)?;
        toml::from_str(&contents)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))
    }
}

#[pyfunction]
pub fn run_optimizer(
    py: Python,
    iterations: numpy::PyReadonlyArray1<u64>,
    all_positions: numpy::PyReadonlyArray4<f32>,
    settings: &Settings,
) -> PyResult<OptimizationResult> {
    let n_agents = all_positions.shape()[1];

    match settings.optimization.borrow(py).deref() {
        OptimizationMethod::DifferentialEvolution(de) => {
            let oinfs = settings.generate_optimization_infos(py, n_agents)?;
            let OptimizationInfos {
                bounds_lower,
                bounds_upper,
                initial_values,
                parameter_infos: _,
                constants: _,
                constant_infos: _,
            } = oinfs;

            let bounds =
                numpy::ndarray::Array2::from_shape_fn((bounds_lower.len(), 2), |(i, j)| {
                    if j == 0 {
                        bounds_lower[i]
                    } else {
                        bounds_upper[i]
                    }
                });
            let locals = pyo3::types::PyDict::new(py);

            // Required
            locals.set_item("bounds", bounds.to_pyarray(py))?;
            locals.set_item("x0", initial_values.into_pyobject(py)?)?;
            locals.set_item("positions_all", all_positions.as_any())?;
            locals.set_item("iterations", iterations.as_any())?;
            locals.set_item("settings", settings.clone().into_pyobject(py)?)?;

            // Optional
            locals.set_item("optimization", de.clone().into_pyobject(py)?)?;

            py.run(
                pyo3::ffi::c_str!(
                    r#"
import scipy as sp
from cr_mech_coli.crm_fit import predict_calculate_cost

args = (positions_all, iterations, settings)

res = sp.optimize.differential_evolution(
    predict_calculate_cost,
    bounds=bounds,
    x0=x0,
    args=args,
    workers=14,
    updating="deferred",
    maxiter=optimization.max_iter,
    disp=True,
    tol=optimization.tol,
    recombination=optimization.recombination,
    popsize=optimization.pop_size,
    polish=optimization.polish,
    rng=optimization.seed,
)
"#
                ),
                None,
                Some(&locals),
            )?;
            let res = locals.get_item("res")?.unwrap();
            println!("something");
            let params: Vec<f32> = res.get_item("x")?.extract()?;
            let cost: f32 = res.get_item("fun")?.extract()?;
            let success: Option<bool> = res.get_item("success").ok().and_then(|x| x.extract().ok());
            let neval: Option<usize> = res.get_item("nfev").ok().and_then(|x| x.extract().ok());
            let niter: Option<usize> = res.get_item("nit").ok().and_then(|x| x.extract().ok());
            Ok(OptimizationResult {
                params,
                cost,
                success,
                neval,
                niter,
            })
        }
        OptimizationMethod::LatinHypercube(lhs) => todo!(),
    }
}
