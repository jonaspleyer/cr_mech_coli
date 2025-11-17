use crate::{
    AgentSettings, Configuration, GrowthRateSetter, PhysInt, PhysicalInteraction,
    RodMechanicsSettings, SpringLengthThresholdSetter, MICRO_METRE, MINUTE,
};
use approx::AbsDiffEq;
use cellular_raza::prelude::MorsePotentialF32;
use pyo3::types::PyDict;
use pyo3::{prelude::*, types::PyString};

use serde::{Deserialize, Serialize};

/// Contain s all parameters and configuration valuese of the crm_multilayer script
#[pyclass(get_all, set_all)]
#[derive(Clone, Debug, Deserialize, Serialize, AbsDiffEq)]
#[approx(epsilon_type = f32)]
pub struct MultilayerConfig {
    /// Contains base configuration. See :class:`Configuration`
    #[approx(map = |b| Python::attach(|py| Some(crate::crm_fit::get_inner(b, py))))]
    pub config: Py<Configuration>,
    /// Contains settings for the Agents of the simulation. See :class:`AgentSettings`
    #[approx(map = |b| Python::attach(|py| Some(crate::crm_fit::get_inner(b, py))))]
    pub agent_settings: Py<AgentSettings>,
    /// Random seed for position generation
    #[approx(equal)]
    pub rng_seed: u64,
    /// Padding of the domain for the position generation algorithm
    #[approx(into_iter)]
    pub dx: [f32; 2],
    /// Controls how much positions are randomized in the beginning of the simulation
    pub randomize_positions: f32,
    /// Number of vertices to use per agent
    #[approx(equal)]
    pub n_vertices: usize,
}

impl PartialEq for MultilayerConfig {
    fn eq(&self, other: &Self) -> bool {
        let MultilayerConfig {
            config,
            agent_settings,
            rng_seed,
            dx,
            randomize_positions,
            n_vertices,
        } = &self;
        Python::attach(|py| {
            use core::ops::Deref;
            config.borrow(py).deref().eq(&other.config.borrow(py))
                && agent_settings
                    .borrow(py)
                    .deref()
                    .eq(&other.agent_settings.borrow(py))
                && rng_seed.eq(&other.rng_seed)
                && dx.eq(&other.dx)
                && randomize_positions.eq(&other.randomize_positions)
                && n_vertices.eq(&other.n_vertices)
        })
    }
}

#[pymethods]
impl MultilayerConfig {
    /// Clones the current MultilayerConfig with new optional keyword arguments
    ///
    /// Args:
    ///     self (MultilayerConfig): Reference to the object itself.
    ///     kwds (dict): Keyword arguments for the new :class:`MultilayerConfig`.
    ///
    #[pyo3(signature = (**kwds))]
    pub fn clone_with_args(&self, py: Python, kwds: Option<&Bound<PyDict>>) -> PyResult<Py<Self>> {
        let new_me: Py<Self> = Py::new(py, self.clone())?;
        if let Some(kwds) = kwds {
            for (key, value) in kwds.iter() {
                let key: Py<PyString> = key.extract()?;
                new_me.setattr(py, &key, value)?;
            }
        }
        Ok(new_me)
    }

    /// Creates a new :class:`MultilayerConfig`
    #[new]
    #[pyo3(signature = (**kwds))]
    pub fn new(py: Python, kwds: Option<&Bound<PyDict>>) -> PyResult<Py<Self>> {
        let n_vertices = 8;
        let config = Py::new(
            py,
            Configuration {
                gel_pressure: 0.01,
                ..Default::default()
            },
        )?;
        let new_self = Py::new(
            py,
            Self {
                config,
                agent_settings: Py::new(
                    py,
                    AgentSettings {
                        mechanics: Py::new(
                            py,
                            RodMechanicsSettings {
                                pos: nalgebra::MatrixXx3::zeros(n_vertices),
                                vel: nalgebra::MatrixXx3::zeros(n_vertices),
                                diffusion_constant: 0.1 * MICRO_METRE.powf(2.0) / MINUTE,
                                spring_tension: 10.0 / MINUTE.powf(2.0),
                                rigidity: 6.0 * MICRO_METRE / MINUTE.powf(2.0),
                                spring_length: 3.0 * MICRO_METRE,
                                damping: 1.5 / MINUTE,
                            },
                        )?,
                        interaction: Py::new(
                            py,
                            PhysicalInteraction(
                                PhysInt::MorsePotentialF32(MorsePotentialF32 {
                                    radius: 3.0 * MICRO_METRE,
                                    potential_stiffness: 0.5 / MICRO_METRE,
                                    cutoff: 8.0 * MICRO_METRE,
                                    strength: 0.1 * MICRO_METRE.powf(2.0) / MINUTE.powf(2.0),
                                }),
                                0,
                            ),
                        )?,
                        growth_rate: 0.1 * MICRO_METRE / MINUTE,
                        growth_rate_setter: Py::new(
                            py,
                            GrowthRateSetter::NormalDistr {
                                mean: 0.1 * MICRO_METRE / MINUTE,
                                std: 0.02 * MICRO_METRE / MINUTE,
                            },
                        )?,
                        spring_length_threshold: 6. * MICRO_METRE,
                        spring_length_threshold_setter: Py::new(
                            py,
                            SpringLengthThresholdSetter::NormalDistr {
                                mean: 6. * MICRO_METRE,
                                std: 0.0,
                            },
                        )?,
                        neighbor_reduction: Some((16, 1.)),
                    },
                )?,
                rng_seed: 0,
                dx: [
                    Configuration::default().domain_size[0] / 2.2,
                    Configuration::default().domain_size[1] / 2.2,
                ],
                randomize_positions: 0.05,
                n_vertices: 8,
            },
        )?;
        if let Some(kwds) = kwds {
            for (key, value) in kwds.iter() {
                let key: Py<PyString> = key.extract()?;
                new_self.setattr(py, &key, value)?;
            }
        }
        Ok(new_self)
    }

    /// Converts the :class:`MultilayerConfig` into a toml string.
    pub fn to_toml_string(&self) -> PyResult<String> {
        toml::to_string(&self).map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))
    }

    /// Saves the :class:`MultilayerConfig` to the given file.
    /// This function will fail if the file already exists.
    pub fn to_toml_file(&self, filename: std::path::PathBuf) -> PyResult<()> {
        use std::io::prelude::*;
        let toml_string = self.to_toml_string()?;
        let mut file = std::fs::File::create_new(filename)?;
        file.write_all(toml_string.as_bytes())?;
        Ok(())
    }

    /// Loads the :class:`MultilayerConfig` from the file at the given path.
    #[staticmethod]
    pub fn load_from_toml_file(path: std::path::PathBuf) -> PyResult<Self> {
        let contents = std::fs::read_to_string(path)?;
        Self::load_from_toml_str(&contents)
    }

    /// Loads the :class:`MultilayerConfig` from the given string.
    #[staticmethod]
    pub fn load_from_toml_str(input: &str) -> PyResult<Self> {
        toml::from_str(input).map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))
    }

    fn approx_eq(&self, other: &Self) -> bool {
        AbsDiffEq::abs_diff_eq(&self, &other, f32::EPSILON)
    }
}

/// A Python module implemented in Rust.
pub fn crm_multilayer_rs(py: Python) -> PyResult<Bound<PyModule>> {
    let m = PyModule::new(py, "cr_mech_coli.crm_multilayer.crm_multilayer_rs")?;
    m.add_class::<MultilayerConfig>()?;
    Ok(m)
}
