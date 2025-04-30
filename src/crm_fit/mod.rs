use core::f32;

use approx::AbsDiffEq;
use cellular_raza::prelude::{MiePotentialF32, MorsePotentialF32, RodInteraction, StorageOption};
use pyo3::{prelude::*, IntoPyObjectExt};
use serde::{Deserialize, Serialize};

use crate::{PhysInt, PhysicalInteraction};

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

#[pymethods]
impl SampledFloat {
    #[new]
    #[pyo3(signature = (min, max, initial, individual=false))]
    fn new(min: f32, max: f32, initial: f32, individual: Option<bool>) -> Self {
        Self {
            min,
            max,
            initial,
            individual,
        }
    }
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
    /// TODO
    #[serde(untagged)]
    #[approx(into_iter)]
    List(Vec<f32>),
}

fn parameter_from_obj(obj: &Bound<PyAny>) -> PyResult<Parameter> {
    if let Ok(value) = obj.extract() {
        Ok(Parameter::Float(value))
    } else if let Ok(value) = obj.extract() {
        Ok(Parameter::SampledFloat(value))
    } else if let Ok(value) = obj.extract() {
        Ok(Parameter::List(value))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Cannot convert object to SampledFloat",
        ))
    }
}

/// TODO
#[pyclass(get_all, module = "cr_mech_coli.crm_fit")]
#[derive(Clone, Debug, Serialize, Deserialize, AbsDiffEq, PartialEq)]
pub struct Parameters {
    /// TODO
    radius: Parameter,
    /// TODO
    rigidity: Parameter,
    /// TODO
    spring_tension: Parameter,
    /// TODO
    damping: Parameter,
    /// TODO
    strength: Parameter,
    /// TODO
    potential_type: PotentialType,
    /// TODO
    growth_rate: Parameter,
}

macro_rules! impl_setters(
    (@single $struct_name:ident $name:ident $setter:ident) => {
        #[pymethods]
        impl $struct_name {
            #[setter]
            fn $setter (&mut self, obj: &Bound<PyAny>) -> PyResult<()> {
                let param = parameter_from_obj(obj)?;
                self.$name = param;
                Ok(())
            }
        }
    };
    ($struct_name:ident; $($name:ident $setter:ident;)*) => {
        $(impl_setters!{@single $struct_name $name $setter})*
    };
);

impl_setters!(
    Parameters;
    radius set_radius;
    rigidity set_rigidity;
    spring_tension set_spring_tension;
    damping set_damping;
    strength set_strength;
    growth_rate set_growth_rate;
);

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
    pub en: Parameter,
    /// TODO
    pub em: Parameter,
    /// TODO
    pub bound: f32,
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
    fn __reduce__<'py>(
        &'py self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        py.run(
            &std::ffi::CString::new("from cr_mech_coli.crm_fit.crm_fit_rs import PotentialType")?,
            None,
            None,
        )
        .unwrap();
        // py.run_bound("from crm_fit import deserialize_potential_type", None, None)
        //     .unwrap();
        let deserialize = py.eval(
            &std::ffi::CString::new("PotentialType.deserialize")?,
            None,
            None,
        )?;
        let data = serde_pickle::to_vec(&self, Default::default()).unwrap();
        Ok((
            deserialize.into_pyobject_or_pyerr(py)?.into_any(),
            (data,).into_pyobject_or_pyerr(py)?.into_any(),
        ))
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
    /// Recombination value of the differential evolution algorithm
    #[serde(default = "default_recombination")]
    pub recombination: f32,
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

const fn default_recombination() -> f32 {
    0.3
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
    #[approx(into_iter)]
    pub domain_size: [f32; 2],
    /// Number of voxels to dissect the domain into
    #[approx(equal)]
    #[serde(default = "default_n_voxels")]
    pub n_voxels: [core::num::NonZeroUsize; 2],
    /// Random initial seed
    #[approx(equal)]
    pub rng_seed: u64,
    /// Cutoff after which the physical interaction is identically zero
    pub cutoff: f32,
    /// Number of vertices to use for discretization of agents
    #[approx(equal)]
    pub n_vertices: core::num::NonZeroUsize,
    /// Number of save points which are not initial and final time point
    #[approx(equal)]
    #[serde(default = "default_n_saves")]
    pub n_saves: usize,
}

const fn default_n_voxels() -> [core::num::NonZeroUsize; 2] {
    [unsafe { core::num::NonZeroUsize::new_unchecked(1) }; 2]
}

const fn default_n_saves() -> usize {
    0
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
    pub fn __reduce__<'py>(
        &'py self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        py.run(
            &std::ffi::CString::new("from cr_mech_coli.crm_fit.crm_fit_rs import Settings")?,
            None,
            None,
        )?;
        // py.run_bound("from crm_fit import deserialize_potential_type", None, None)
        //     .unwrap();
        let deserialize = py.eval(&std::ffi::CString::new("Settings.deserialize")?, None, None)?;
        let data = serde_pickle::to_vec(&self, Default::default()).unwrap();
        Ok((
            deserialize.into_pyobject_or_pyerr(py)?.into_any(),
            (data,).into_pyobject_or_pyerr(py)?.into_any(),
        ))
    }

    /// Converts the settings provided to a :class:`Configuration` object required to run the
    /// simulation
    pub fn to_config(&self) -> PyResult<crate::Configuration> {
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
                    n_vertices,
                    n_saves,
                },
            parameters,
            optimization,
        } = self.clone();
        Ok(crate::Configuration {
            domain_height: self.domain_height(),
            n_threads: 1.try_into().unwrap(),
            t0: 0.0,
            dt,
            t_max,
            n_saves,
            show_progressbar: false,
            domain_size,
            n_voxels: [n_voxels[0].get(), n_voxels[1].get()],
            rng_seed,
            gravity: 0.,
            surface_friction: 0.,
            surface_friction_distance: 1.,
            storage_options: vec![StorageOption::Memory],
            storage_location: std::path::PathBuf::new(),
        })
    }

    /// Creates a list of lower and upper bounds for the sampled parameters
    #[allow(unused)]
    pub fn generate_optimization_infos(
        &self,
        n_agents: usize,
    ) -> (
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<(String, String, String)>,
        Vec<f32>,
        Vec<(String, String, String)>,
    ) {
        let mut param_space_dim = 0;

        #[allow(unused)]
        let Parameters {
            radius,
            rigidity,
            spring_tension,
            damping,
            strength,
            potential_type,
            growth_rate,
        } = &self.parameters;

        let mut bounds_lower = Vec::new();
        let mut bounds_upper = Vec::new();
        let mut initial_values = Vec::new();
        let mut infos = Vec::new();
        let mut constants = Vec::new();
        let mut constant_infos = Vec::new();
        macro_rules! append_infos_bounds(
            ($var:expr, $var_name:expr, $units:expr, $symbol:expr) => {
                match &$var {
                    Parameter::SampledFloat(SampledFloat {
                        min,
                        max,
                        initial,
                        individual,
                    }) => {
                        if individual.is_none() || individual == &Some(false) {
                            bounds_lower.push(min.clone());
                            bounds_upper.push(max.clone());
                            param_space_dim += 1;
                            infos.push((
                                $var_name.to_string(),
                                $units.to_string(),
                                $symbol.to_string(),
                            ));
                            initial_values.push(initial.clone());
                        } else {
                            bounds_lower.extend(vec![min.clone(); n_agents]);
                            bounds_upper.extend(vec![max.clone(); n_agents]);
                            param_space_dim += n_agents;
                            infos.extend(
                                (0..n_agents)
                                    .map(|i| (
                                        format!("{} {}", $var_name, i),
                                        $units.to_string(),
                                        format!("{}_{{{}}}", $symbol, i),
                                    ))
                            );
                            initial_values.extend(vec![initial.clone(); n_agents]);
                        }
                    },
                    Parameter::Float(c) => {
                        constants.push(*c);
                        constant_infos.push((
                            $var_name.to_string(),
                            $units.to_string(),
                            $symbol.to_string(),
                        ));
                    },
                    Parameter::List(list) => {
                        constants.extend(list);
                        constant_infos.push((
                            $var_name.to_string(),
                            $units.to_string(),
                            $symbol.to_string(),
                        ));
                    },
                }
            }
        );
        append_infos_bounds!(radius, "Radius", "\\SI{}{\\micro\\metre}", "r");
        append_infos_bounds!(
            rigidity,
            "Rigidity",
            "\\SI{}{\\micro\\metre\\per\\min}",
            "\\kappa"
        );
        append_infos_bounds!(
            spring_tension,
            "Spring Tension",
            "\\SI{}{\\per\\min^2}",
            "\\gamma"
        );
        append_infos_bounds!(damping, "Damping", "\\SI{}{\\per\\min}", "\\lambda");
        append_infos_bounds!(
            strength,
            "Strength",
            "\\SI{}{\\micro\\metre^2\\per\\min^2}",
            "C"
        );
        append_infos_bounds!(growth_rate, "Growth Rate", "\\SI{}{\\per\\min}", "\\mu");
        match potential_type {
            PotentialType::Mie(mie) => {
                let en = mie.en.clone();
                let em = mie.em.clone();
                append_infos_bounds!(en, "Exponent n", "1", "n");
                append_infos_bounds!(em, "Exponent m", "1", "m");
            }
            PotentialType::Morse(morse) => append_infos_bounds!(
                &morse.potential_stiffness,
                "Potential Stiffness",
                "\\SI{}{\\micro\\metre}",
                "\\lambda"
            ),
        }

        (
            bounds_lower,
            bounds_upper,
            initial_values,
            infos,
            constants,
            constant_infos,
        )
    }

    /// TODO
    pub fn predict(
        &self,
        parameters: Vec<f32>,
        positions: numpy::PyReadonlyArray3<f32>,
    ) -> PyResult<crate::CellContainer> {
        let config = self.to_config()?;
        let mut positions = positions.as_array().to_owned();

        // If the positions do not have dimension (?,?,3), we bring them to this dimension
        if positions.shape()[2] != 3 {
            let mut new_positions = numpy::ndarray::Array3::<f32>::zeros((
                positions.shape()[0],
                positions.shape()[1],
                3,
            ));
            new_positions
                .slice_mut(numpy::ndarray::s![.., .., ..2])
                .assign(&positions.slice(numpy::ndarray::s![.., .., ..2]));
            use core::ops::AddAssign;
            new_positions
                .slice_mut(numpy::ndarray::s![.., .., 2])
                .add_assign(self.domain_height() / 2.0);
            positions = new_positions;
        }
        let n_agents = positions.shape()[0];

        let Parameters {
            radius,
            rigidity,
            spring_tension,
            damping,
            strength,
            potential_type,
            growth_rate,
        } = &self.parameters;

        let mut param_counter = 0;
        macro_rules! check_parameter(
            ($var:expr) => {
                match $var {
                    // Fixed
                    Parameter::Float(value) => {
                        vec![value.clone(); n_agents]
                    },
                    #[allow(unused)]
                    Parameter::SampledFloat(SampledFloat {
                        min,
                        max,
                        initial: _,
                        individual,
                    }) => {
                        // Sampled-Individual
                        if individual == &Some(true) {
                            let res = parameters[param_counter..param_counter+n_agents]
                                .to_vec();
                            param_counter += n_agents;
                            res
                        // Sampled-Single
                        } else {
                            let res = vec![parameters[param_counter]; n_agents];
                            param_counter += 1;
                            res
                        }
                    },
                    Parameter::List(list) => list.clone(),
                }
            };
        );

        let (radius, rigidity, spring_tension, damping, strength, growth_rate) = (
            check_parameter!(radius),
            check_parameter!(rigidity),
            check_parameter!(spring_tension),
            check_parameter!(damping),
            check_parameter!(strength),
            check_parameter!(growth_rate),
        );

        // Now configure potential type
        let interaction: Vec<_> = match potential_type {
            PotentialType::Mie(Mie { en, em, bound }) => {
                let en = check_parameter!(en);
                let em = check_parameter!(em);
                en.into_iter()
                    .zip(em)
                    .enumerate()
                    .map(|(n, (en, em))| {
                        RodInteraction(PhysicalInteraction(
                            PhysInt::MiePotentialF32(MiePotentialF32 {
                                en,
                                em,
                                strength: strength[n],
                                radius: radius[n],
                                bound: *bound,
                                cutoff: self.constants.cutoff,
                            }),
                            0,
                        ))
                    })
                    .collect()
            }
            PotentialType::Morse(Morse {
                potential_stiffness,
            }) => {
                let potential_stiffness = check_parameter!(potential_stiffness);
                potential_stiffness
                    .into_iter()
                    .enumerate()
                    .map(|(n, potential_stiffness)| {
                        RodInteraction(PhysicalInteraction(
                            PhysInt::MorsePotentialF32(MorsePotentialF32 {
                                strength: strength[n],
                                radius: radius[n],
                                potential_stiffness,
                                cutoff: self.constants.cutoff,
                            }),
                            0,
                        ))
                    })
                    .collect()
            }
        };

        let pos_to_spring_length = |pos: &nalgebra::MatrixXx3<f32>| -> f32 {
            let mut res = 0.0;
            for i in 0..pos.nrows() - 1 {
                res += ((pos[(i + 1, 0)] - pos[(i, 0)]).powf(2.0)
                    + (pos[(i + 1, 1)] - pos[(i, 1)]).powf(2.0))
                .sqrt();
            }
            res / (self.constants.n_vertices.get() - 1) as f32
        };

        let agents = positions
            .axis_iter(numpy::ndarray::Axis(0))
            .enumerate()
            .map(|(n, pos)| {
                let pos = nalgebra::Matrix3xX::<f32>::from_iterator(
                    self.constants.n_vertices.get(),
                    pos.iter().copied(),
                );
                let spring_length = pos_to_spring_length(&pos.transpose());
                crate::RodAgent {
                    mechanics: cellular_raza::prelude::RodMechanics {
                        pos: pos.transpose(),
                        vel: nalgebra::MatrixXx3::zeros(self.constants.n_vertices.get()),
                        diffusion_constant: 0.0,
                        spring_tension: spring_tension[n],
                        rigidity: rigidity[n],
                        spring_length,
                        damping: damping[n],
                    },
                    interaction: interaction[n].clone(),
                    growth_rate: growth_rate[n],
                    spring_length_threshold: f32::INFINITY,
                    neighbor_reduction: None,
                }
            })
            .collect();
        Ok(crate::run_simulation_with_agents(&config, agents)?)
    }

    /// Formats the object
    pub fn __repr__(&self) -> String {
        format!("{self:#?}")
    }
}

/// A Python module implemented in Rust.
pub fn crm_fit_rs(py: Python) -> PyResult<Bound<PyModule>> {
    let m = PyModule::new(py, "crm_fit_rs")?;
    m.add_class::<SampledFloat>()?;
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

    fn generate_test_settings() -> (Settings, String) {
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
                domain_size: [100.0; 2],
                n_voxels: [1.try_into().unwrap(); 2],
                rng_seed: 0,
                cutoff: 20.0,
                n_vertices: 8.try_into().unwrap(),
                n_saves: 0,
            },
            parameters: Parameters {
                radius: Parameter::SampledFloat(SampledFloat {
                    min: 3.0,
                    max: 6.0,
                    initial: 4.5,
                    individual: Some(true),
                }),
                rigidity: Parameter::Float(8.0),
                spring_tension: Parameter::Float(1.0),
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
                growth_rate: Parameter::SampledFloat(SampledFloat {
                    min: 0.0,
                    max: 10.0,
                    initial: 1.0,
                    individual: None,
                }),
            },
            optimization: Optimization {
                seed: 0,
                tol: 1e-3,
                max_iter: default_max_iter(),
                pop_size: default_pop_size(),
                recombination: default_recombination(),
            },
        };
        let toml_string = "
[constants]
t_max=100.0
dt=0.005
domain_size=[100, 100]
n_voxels=[1, 1]
rng_seed=0
cutoff=20.0
n_vertices=8

[parameters]
radius = { min = 3.0, max=6.0, initial=4.5, individual=true }
rigidity = 8.0
spring_tension = 1.0
damping = { min=0.6, max=2.5, initial=1.5 }
strength = { min=1.0, max=4.5, initial=1.0 }
growth_rate = { min=0.0, max=10.0, initial=1.0 }

[parameters.potential_type.Mie]
en = { min=0.2, max=25.0, initial=6.0, individual=false}
em = { min=0.2, max=25.0, initial=5.5}
bound = 8.0

[optimization]
seed = 0
tol = 1e-3
"
        .to_string();
        (settings1, toml_string)
    }

    #[test]
    fn test_parsing_toml() {
        let (settings1, toml_string) = generate_test_settings();
        let settings = Settings::from_toml_string(&toml_string).unwrap();
        approx::assert_abs_diff_eq!(settings1, settings);
    }

    #[test]
    fn test_bound_generation() {
        let (settings, _) = generate_test_settings();

        for n_agents in 1..10 {
            let (lower, upper, _, _, _, _) = settings.generate_optimization_infos(n_agents);
            assert_eq!(lower.len(), n_agents + 5);
            assert_eq!(upper.len(), n_agents + 5);
        }
    }
}
