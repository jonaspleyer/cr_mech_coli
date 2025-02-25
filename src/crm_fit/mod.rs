use approx_derive::AbsDiffEq;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// TODO
#[pyclass(get_all, set_all, module = "cr_mech_coli.crm_fit")]
#[derive(Clone, Debug, Serialize, Deserialize, AbsDiffEq, PartialEq)]
#[approx(epsilon_type = f32)]
pub struct SampledFloat {
    /// TODO
    pub min: f32,
    /// TODO
    pub max: f32,
    /// TODO
    pub initial: f32,
    /// TODO
    #[approx(equal)]
    pub individual: Option<bool>,
}

/// TODO
#[pyclass(get_all, set_all, module = "cr_mech_coli.crm_fit")]
#[derive(Clone, Debug, Serialize, Deserialize, AbsDiffEq, PartialEq)]
#[approx(epsilon_type = f32)]
pub enum Parameter {
    /// TODO
    #[serde(untagged)]
    SampledFloat(SampledFloat),
    /// TODO
    #[serde(untagged)]
    Float(f32),
}

/// TODO
#[pyclass(get_all, set_all, module = "cr_mech_coli.crm_fit")]
#[derive(Clone, Debug, Serialize, Deserialize, AbsDiffEq, PartialEq)]
pub struct Parameters {
    /// TODO
    radius: Parameter,
    /// TODO
    rigidity: Parameter,
    /// TODO
    damping: Parameter,
    /// TODO
    strength: Parameter,
    // TODO
    potential_type: PotentialType,
}

/// TODO
#[pyclass(get_all, set_all, module = "cr_mech_coli.crm_fit")]
#[derive(Clone, Debug, Serialize, Deserialize, AbsDiffEq, PartialEq)]
#[approx(epsilon_type = f32)]
pub struct Morse {
    /// TODO
    potential_stiffness: Parameter,
}

/// TODO
#[pyclass(get_all, set_all, module = "cr_mech_coli.crm_fit")]
#[derive(Clone, Debug, Serialize, Deserialize, AbsDiffEq, PartialEq)]
#[approx(epsilon_type = f32)]
pub struct Mie {
    /// TODO
    en: Parameter,
    /// TODO
    em: Parameter,
    /// TODO
    bound: f32,
}

/// TODO
#[pyclass(get_all, set_all, module = "cr_mech_coli.crm_fit")]
#[derive(Clone, Debug, Serialize, Deserialize, AbsDiffEq, PartialEq)]
pub enum PotentialType {
    /// TODO
    Mie(Mie),
    /// TODO
    Morse(Morse),
}

#[pymethods]
impl PotentialType {
    // Reconstructs a interaction potential
    // pub fn reconstruct_potential(&self, radius: f32, strength: f32, cutoff: f32) {}

    /// Formats the object
    pub fn to_short_string(&self) -> String {
        match self {
            PotentialType::Mie(_) => "mie".to_string(),
            PotentialType::Morse(_) => "morse".to_string(),
        }
    }

    /// Helper method for :func:`~PotentialType.__reduce__`
    #[staticmethod]
    fn deserialize(data: Vec<u8>) -> Self {
        serde_pickle::from_slice(&data, Default::default()).unwrap()
    }

    /// Used to pickle the :class:`PotentialType`
    fn __reduce__(&self) -> (PyObject, PyObject) {
        Python::with_gil(|py| {
            py.run_bound(
                "from cr_mech_coli.crm_fit.crm_fit_rs import PotentialType",
                None,
                None,
            )
            .unwrap();
            // py.run_bound("from crm_fit import deserialize_potential_type", None, None)
            //     .unwrap();
            let deserialize = py
                .eval_bound("PotentialType.deserialize", None, None)
                .unwrap();
            let data = serde_pickle::to_vec(&self, Default::default()).unwrap();
            (deserialize.to_object(py), (data,).to_object(py))
        })
    }
}

/// TODO
#[pyclass(get_all, set_all, module = "cr_mech_coli.crm_fit")]
#[derive(Clone, Debug, Serialize, Deserialize, AbsDiffEq, PartialEq)]
#[approx(epsilon_type = f32)]
pub struct Optimization {
    /// Initial seed of the differential evolution algorithm
    #[serde(default)]
    #[approx(equal)]
    pub seed: u64,
    /// Tolerance of the differential evolution algorithm
    #[serde(default = "default_tol")]
    pub tol: f32,
    /// Maximum iterations of the differential evolution algorithm
    #[serde(default = "default_max_iter")]
    #[approx(equal)]
    pub max_iter: usize,
    /// Population size for each iteration
    #[serde(default = "default_pop_size")]
    #[approx(equal)]
    pub pop_size: usize,
}

const fn default_tol() -> f32 {
    1e-4
}

const fn default_max_iter() -> usize {
    50
}

const fn default_pop_size() -> usize {
    100
}

/// Contains all constants of the numerical simulation
#[pyclass(get_all, set_all, module = "cr_mech_coli.crm_fit")]
#[derive(Clone, Debug, Serialize, Deserialize, AbsDiffEq, PartialEq)]
pub struct Constants {
    /// Total time from start to finish
    pub t_max: f32,
    /// Time increment used to solve equations
    pub dt: f32,
    /// Size of the domain
    pub domain_size: f32,
    /// Number of voxels to dissect the domain into
    #[approx(equal)]
    pub n_voxels: core::num::NonZeroUsize,
    /// Random initial seed
    #[approx(equal)]
    pub rng_seed: u64,
    /// Cutoff after which the physical interaction is identically zero
    pub cutoff: f32,
    /// Conversion between pixels and micron.
    pub pixel_per_micron: f32,
}

/// Contains all settings required to fit the model to images
#[pyclass(get_all, set_all, module = "cr_mech_coli.crm_fit")]
#[derive(Clone, Debug, Serialize, Deserialize, AbsDiffEq, PartialEq)]
#[approx(epsilon_type = f32)]
pub struct Settings {
    /// See :class:`Constants`
    pub constants: Constants,
    /// See :class:`Parameters`
    pub parameters: Parameters,
    /// See :class:`OptimizationParameters`
    pub optimization: Optimization,
}

#[pymethods]
impl Settings {
    /// Creates a :class:`Settings` from a given toml string.
    /// See also :func:`~Settings.from_toml_string`.
    #[staticmethod]
    pub fn from_toml(toml_filename: std::path::PathBuf) -> PyResult<Self> {
        let content = std::fs::read_to_string(toml_filename)?;
        Self::from_toml_string(&content)
    }

    /// Parses the contents of the given string and returns a :class:`Settings` object.
    /// See also :func:`~Settings.from_toml`.
    #[staticmethod]
    pub fn from_toml_string(toml_string: &str) -> PyResult<Self> {
        toml::from_str(toml_string)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))
    }

    /// Creates a toml string from the configuration file
    pub fn to_toml(&self) -> PyResult<String> {
        toml::to_string(&self).map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))
    }

    /// Obtains the domain height
    #[getter]
    pub fn domain_height(&self) -> f32 {
        2.5
    }

    /// Helper method for :func:`~PotentialType.__reduce__`
    #[staticmethod]
    fn deserialize(data: Vec<u8>) -> Self {
        serde_pickle::from_slice(&data, Default::default()).unwrap()
    }

    /// Implements the `__reduce__` method used by pythons pickle protocol.
    pub fn __reduce__(&self) -> (PyObject, PyObject) {
        Python::with_gil(|py| {
            py.run_bound(
                "from cr_mech_coli.crm_fit.crm_fit_rs import Settings",
                None,
                None,
            )
            .unwrap();
            // py.run_bound("from crm_fit import deserialize_potential_type", None, None)
            //     .unwrap();
            let deserialize = py.eval_bound("Settings.deserialize", None, None).unwrap();
            let data = serde_pickle::to_vec(&self, Default::default()).unwrap();
            (deserialize.to_object(py), (data,).to_object(py))
        })
    }

    /// Converts the settings provided to a :class:`Configuration` object required to run the
    /// simulation
    pub fn to_config(&self, n_saves: usize) -> PyResult<crate::Configuration> {
        #[allow(unused)]
        let Self {
            constants:
                Constants {
                    t_max,
                    dt,
                    domain_size,
                    n_voxels,
                    rng_seed,
                    cutoff,
                    pixel_per_micron,
                },
            parameters,
            optimization,
        } = self.clone();
        let save_interval = t_max / n_saves as f32;
        Ok(crate::Configuration {
            domain_height: self.domain_height(),
            n_threads: 1.try_into().unwrap(),
            t0: 0.0,
            dt,
            t_max,
            save_interval,
            show_progressbar: false,
            domain_size,
            n_voxels: n_voxels.get(),
            rng_seed,
        })
    }

    /// Formats the object
    pub fn __repr__(&self) -> String {
        format!("{self:#?}")
    }
}

/// A Python module implemented in Rust.
pub fn crm_fit_rs(py: Python) -> PyResult<Bound<PyModule>> {
    let m = PyModule::new_bound(py, "crm_fit_rs")?;
    m.add_class::<Parameter>()?;
    m.add_class::<Constants>()?;
    m.add_class::<Parameters>()?;
    m.add_class::<Optimization>()?;
    m.add_class::<Settings>()?;
    m.add_class::<PotentialType>()?;
    m.add_class::<PotentialType_Morse>()?;
    m.add_class::<PotentialType_Mie>()?;
    Ok(m)
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_parsing_toml() {
        let potential_type = PotentialType::Mie(Mie {
            en: Parameter::SampledFloat(SampledFloat {
                min: 0.2,
                max: 25.0,
                initial: 6.0,
                individual: Some(false),
            }),
            em: Parameter::SampledFloat(SampledFloat {
                min: 0.2,
                max: 25.0,
                initial: 5.5,
                individual: None,
            }),
            bound: 8.0,
        });
        let settings1 = Settings {
            constants: Constants {
                t_max: 100.0,
                dt: 0.005,
                domain_size: 100.0,
                n_voxels: 1.try_into().unwrap(),
                rng_seed: 0,
                cutoff: 20.0,
                pixel_per_micron: 2.2,
            },
            parameters: Parameters {
                radius: Parameter::SampledFloat(SampledFloat {
                    min: 3.0,
                    max: 6.0,
                    initial: 4.5,
                    individual: Some(true),
                }),
                rigidity: Parameter::Float(8.0),
                damping: Parameter::SampledFloat(SampledFloat {
                    min: 0.6,
                    max: 2.5,
                    initial: 1.5,
                    individual: None,
                }),
                strength: Parameter::SampledFloat(SampledFloat {
                    min: 1.0,
                    max: 4.5,
                    initial: 1.0,
                    individual: None,
                }),
                potential_type,
            },
            optimization: Optimization {
                seed: 0,
                tol: 1e-4,
                max_iter: default_max_iter(),
                pop_size: default_pop_size(),
            },
        };
        let toml_string = "
[constants]
t_max=100.0
dt=0.005
domain_size=100.0
n_voxels=1
rng_seed=0
cutoff=20.0
pixel_per_micron=2.2

[parameters]
radius = { min = 3.0, max=6.0, initial=4.5, individual=true }
rigidity = 8.0
damping = { min=0.6, max=2.5, initial=1.5 }
strength = { min=1.0, max=4.5, initial=1.0 }

[parameters.potential_type.Mie]
en = { min=0.2, max=25.0, initial=6.0, individual=false}
em = { min=0.2, max=25.0, initial=5.5}
bound = 8.0

[optimization]
seed = 0
tol = 1e-4
"
        .to_string();
        let settings: Settings = toml::from_str(&toml_string).unwrap();
        approx::assert_abs_diff_eq!(settings1, settings);
    }

    /* #[test]
    fn test_parsing() {
        let settings = Settings {
            constants: Constants { domain_size: 100.0 },
            parameters: Parameters {
                t_max: Parameter::SampledFloat(SampledFloat {
                    min: 0.0,
                    max: 10.0,
                    initial: 5.0,
                }),
                domain_size: Parameter::Float(100.0),
            },
            optimization: Optimization { workers: None },
        };
        let toml_string = settings.to_toml().unwrap();
        println!("{toml_string}");
    }*/
}
