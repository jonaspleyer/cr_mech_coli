use std::ops::Deref;

use crate::crm_fit::predict::predict_calculate_cost_rs;

use super::settings::*;

use egobox_doe::SamplingMethod;
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

fn lhs_optimization(
    py: Python,
    n_points: usize,
    n_iter: usize,
    bounds: &numpy::ndarray::Array2<f32>,
    iterations_images: &[usize],
    positions_all: numpy::ndarray::ArrayView4<f32>,
    settings: &Settings,
) -> PyResult<Option<(Vec<f32>, f32)>> {
    use kdam::{term::Colorizer, *};
    use rayon::prelude::*;

    let domain_height = settings.domain_height();
    let constants: Constants = settings.constants.extract(py)?;
    let parameter_defs: Parameters = settings.parameters.extract(py)?;
    let config = settings.to_config(py)?;

    let lhs_doe = egobox_doe::Lhs::new(bounds);
    let combinations = lhs_doe.sample(n_points);

    // Initialize progress bar
    kdam::term::init(true);
    let result = kdam::par_tqdm!(
        combinations.axis_iter(ndarray::Axis(0)).into_par_iter(),
        desc = format!("LHS Step {n_iter}").colorize("green"),
        total = n_points
    )
    // Calculate Costs for every sampled parameter point
    .filter_map(|parameters| {
        predict_calculate_cost_rs(
            parameters.to_vec(),
            positions_all,
            domain_height,
            &parameter_defs,
            &constants,
            &config,
            iterations_images,
        )
        .ok()
        .map(|x| (parameters.to_vec(), x))
    })
    .filter(|x| x.1.is_finite())
    .reduce_with(|x, y| if x.1 < y.1 { x } else { y });

    if let Some((_, cost)) = result {
        println!("Final cost: {}", format!("{cost}").colorize("blue"));
    }

    Ok(result)
}

#[pyfunction]
pub fn run_optimizer(
    py: Python,
    iterations: Vec<usize>,
    positions_all: numpy::PyReadonlyArray4<f32>,
    settings: &Settings,
) -> PyResult<OptimizationResult> {
    let n_agents = positions_all.shape()[1];
    let oinfs = settings.generate_optimization_infos(py, n_agents)?;
    let OptimizationInfos {
        bounds_lower,
        bounds_upper,
        initial_values,
        parameter_infos: _,
        constants: _,
        constant_infos: _,
    } = oinfs;

    let bounds = numpy::ndarray::Array2::from_shape_fn((bounds_lower.len(), 2), |(i, j)| {
        if j == 0 {
            bounds_lower[i]
        } else {
            bounds_upper[i]
        }
    });

    let positions_all = positions_all.as_array();
    let domain_height = settings.domain_height();
    let constants: Constants = settings.constants.extract(py)?;
    let parameter_defs: Parameters = settings.parameters.extract(py)?;
    let config = settings.to_config(py)?;

    match settings.optimization.borrow(py).deref() {
        OptimizationMethod::DifferentialEvolution(de) => {
            let locals = pyo3::types::PyDict::new(py);

            // Required
            locals.set_item("bounds", bounds.to_pyarray(py))?;
            locals.set_item("x0", initial_values.into_pyobject(py)?)?;
            locals.set_item("positions_all", positions_all.to_pyarray(py))?;
            locals.set_item("iterations", iterations)?;
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
        OptimizationMethod::LatinHypercube(lhs) => {
            use kdam::*;
            use rayon::prelude::*;

            // Sample the space
            let LatinHypercube { n_points } = lhs;
            let lhs_doe = egobox_doe::Lhs::new(&bounds);
            let combinations = lhs_doe.sample(*n_points);

            // Calculate Costs for every sampled parameter point
            let result = kdam::par_tqdm!(
                combinations.axis_iter(ndarray::Axis(0)).into_par_iter(),
                desc = "Optimization LatinHypercube",
                total = *n_points
            )
            .filter_map(move |parameters| {
                predict_calculate_cost_rs(
                    parameters.to_vec(),
                    positions_all,
                    domain_height,
                    &parameter_defs,
                    &constants,
                    &config,
                    &iterations,
                )
                .ok()
                .map(|x| (parameters.to_vec(), x))
            })
            .filter(|x| x.1.is_finite())
            .reduce_with(|x, y| if x.1 < y.1 { x } else { y });

            // Return Optizmization Result
            if let Some((params, cost)) = result {
                let params = params.to_vec();
                Ok(OptimizationResult {
                    params,
                    cost,
                    success: Some(true),
                    neval: Some(*n_points),
                    niter: None,
                })
            } else {
                Ok(OptimizationResult {
                    params: initial_values.clone(),
                    cost: f32::NAN,
                    success: Some(false),
                    neval: Some(*n_points),
                    niter: None,
                })
            }
        }
    }
}
