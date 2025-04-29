use crate::{AgentSettings, Configuration, PhysInt, PhysicalInteraction, RodMechanicsSettings};
use cellular_raza::prelude::MorsePotentialF32;
use pyo3::types::PyDict;
use pyo3::{prelude::*, types::PyString};

use serde::{Deserialize, Serialize};

/// Micro metre base unit
pub const MICRO_METRE: f32 = 1.0;
/// Minute in base unit
pub const MINUTE: f32 = 1.0;
/// Hour derived from [MINUTE]
pub const HOUR: f32 = 60. * MINUTE;

/// Contain s all parameters and configuration valuese of the crm_multilayer script
#[pyclass(get_all, set_all)]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MultilayerConfig {
    /// Contains base configuration. See :class:`Configuration`
    pub config: Py<Configuration>,
    /// Contains settings for the Agents of the simulation. See :class:`AgentSettings`
    pub agent_settings: Py<AgentSettings>,
    /// Random seed for position generation
    pub rng_seed: u64,
    /// Padding of the domain for the position generation algorithm
    pub dx: [f32; 2],
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
                gravity: 0.01,
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
                        spring_length_threshold: 6. * MICRO_METRE,
                        neighbor_reduction: Some((16, 1.)),
                    },
                )?,
                rng_seed: 0,
                dx: [
                    Configuration::default().domain_size[0] / 2.2,
                    Configuration::default().domain_size[1] / 2.2,
                ],
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
}

/// A Python module implemented in Rust.
pub fn crm_multilayer_rs(py: Python) -> PyResult<Bound<PyModule>> {
    let m = PyModule::new(py, "crm_multilayer_rs")?;
    m.add_class::<MultilayerConfig>()?;
    Ok(m)
}
