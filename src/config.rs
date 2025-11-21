use approx::AbsDiffEq;
use std::{hash::Hasher, num::NonZeroUsize};

use numpy::PyUntypedArrayMethods;
use pyo3::IntoPyObjectExt;
use pyo3::{prelude::*, types::PyString};
use serde::{Deserialize, Serialize};

use crate::agent::{PhysInt, PhysicalInteraction, RodAgent};
use crate::{GrowthRateSetter, SpringLengthThresholdSetter};

use cellular_raza::prelude::{RodInteraction, RodMechanics, StorageOption};

fn serialize_matrixxx3<S>(m: &nalgebra::MatrixXx3<f32>, s: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    let n = nalgebra::DMatrix::<f32>::from_iterator(m.nrows(), m.ncols(), m.iter().copied());
    serde::Serialize::serialize(&n, s)
}

fn deserialize_matrixxx3<'de, D>(de: D) -> Result<nalgebra::MatrixXx3<f32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let n: nalgebra::DMatrix<f32> = serde::Deserialize::deserialize(de)?;
    let m = nalgebra::MatrixXx3::from_iterator(n.nrows(), n.iter().copied());
    Ok(m)
}

/// Contains all settings required to construct :class:`RodMechanics`
#[pyclass]
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, AbsDiffEq)]
#[approx(epsilon_type = f32)]
pub struct RodMechanicsSettings {
    /// The current position
    #[approx(into_iter)]
    #[serde(serialize_with = "serialize_matrixxx3")]
    #[serde(deserialize_with = "deserialize_matrixxx3")]
    pub pos: nalgebra::MatrixXx3<f32>,
    /// The current velocity
    #[approx(into_iter)]
    #[serde(serialize_with = "serialize_matrixxx3")]
    #[serde(deserialize_with = "deserialize_matrixxx3")]
    pub vel: nalgebra::MatrixXx3<f32>,
    /// Controls magnitude of32 stochastic motion
    #[pyo3(get, set)]
    pub diffusion_constant: f32,
    /// Spring tension between individual vertices
    #[pyo3(get, set)]
    pub spring_tension: f32,
    /// Stif32fness at each joint connecting two edges
    #[pyo3(get, set)]
    pub rigidity: f32,
    /// Target spring length
    #[pyo3(get, set)]
    pub spring_length: f32,
    /// Damping constant
    #[pyo3(get, set)]
    pub damping: f32,
}

impl From<RodMechanicsSettings> for RodMechanics<f32, 3> {
    fn from(value: RodMechanicsSettings) -> Self {
        let RodMechanicsSettings {
            pos,
            vel,
            diffusion_constant,
            spring_tension,
            rigidity,
            spring_length,
            damping,
        } = value;
        RodMechanics {
            pos,
            vel,
            diffusion_constant,
            spring_tension,
            rigidity,
            spring_length,
            damping,
        }
    }
}

#[pymethods]
impl RodMechanicsSettings {
    fn __repr__(&self) -> String {
        format!("{:#?}", self)
    }

    #[getter]
    fn pos<'a>(&'a self, py: Python<'a>) -> Bound<'a, numpy::PyArray2<f32>> {
        use numpy::ToPyArray;
        let nrows = self.pos.nrows();
        let new_array =
            numpy::nalgebra::MatrixXx3::from_iterator(nrows, self.pos.iter().map(Clone::clone));
        new_array.to_pyarray(py)
    }

    #[setter]
    fn set_pos<'a>(&'a mut self, pos: Bound<'a, numpy::PyArray2<f32>>) -> pyo3::PyResult<()> {
        use numpy::PyArrayMethods;
        let nrows = pos.shape()[0];
        self.pos = nalgebra::MatrixXx3::<f32>::from_iterator(nrows, pos.to_vec()?);
        Ok(())
    }

    #[getter]
    fn vel<'a>(&'a self, py: Python<'a>) -> Bound<'a, numpy::PyArray2<f32>> {
        use numpy::ToPyArray;
        let new_array = numpy::nalgebra::MatrixXx3::<f32>::from_iterator(
            self.vel.nrows(),
            self.vel.iter().map(Clone::clone),
        );
        new_array.to_pyarray(py)
    }

    #[setter]
    fn set_vel<'a>(&'a mut self, pos: Bound<'a, numpy::PyArray2<f32>>) -> pyo3::PyResult<()> {
        use numpy::PyArrayMethods;
        let nrows = pos.shape()[0];
        self.vel = nalgebra::MatrixXx3::<f32>::from_iterator(nrows, pos.to_vec()?);
        Ok(())
    }
}

impl Default for RodMechanicsSettings {
    fn default() -> Self {
        RodMechanicsSettings {
            pos: nalgebra::MatrixXx3::zeros(8),
            vel: nalgebra::MatrixXx3::zeros(8),
            diffusion_constant: 0.0, // MICROMETRE^2 / MIN^2
            spring_tension: 1.0,     // 1 / MIN
            rigidity: 2.0,
            spring_length: 3.0, // MICROMETRE
            damping: 1.0,       // 1/MIN
        }
    }
}

/// Contains settings needed to specify properties of the :class:`RodAgent`
#[pyclass(get_all, set_all, mapping)]
#[derive(Clone, Debug, Deserialize, Serialize, AbsDiffEq)]
#[approx(epsilon_type = f32)]
pub struct AgentSettings {
    /// Settings for the mechanics part of :class:`RodAgent`. See also :class:`RodMechanicsSettings`.
    #[approx(map = |b| Python::attach(|py| Some(crate::crm_fit::get_inner(b, py))))]
    pub mechanics: Py<RodMechanicsSettings>,
    /// Settings for the interaction part of :class:`RodAgent`. See also :class:`MorsePotentialF32`.
    #[approx(map = |b| Python::attach(|py| Some(crate::crm_fit::get_inner(b, py))))]
    pub interaction: Py<PhysicalInteraction>,
    /// Rate with which the length of the bacterium grows
    pub growth_rate: f32,
    /// See :class:`RodAgent`
    #[approx(map = |b| Python::attach(|py| Some(crate::crm_fit::get_inner(b, py))))]
    pub growth_rate_setter: Py<GrowthRateSetter>,
    /// Threshold when the bacterium divides
    pub spring_length_threshold: f32,
    /// Sets the spring_length_threshold after a division event
    #[approx(map = |b| Python::attach(|py| Some(crate::crm_fit::get_inner(b, py))))]
    pub spring_length_threshold_setter: Py<SpringLengthThresholdSetter>,
    /// Reduces the growth rate with multiplier $((max - N)/max)^q $
    #[approx(map = |x: &Option<(usize, f32)>| x.map(|(x, y)| (x as f32, y)))]
    #[approx(epsilon_map = |x| (x, x,))]
    pub neighbor_reduction: Option<(usize, f32)>,
}

impl PartialEq for AgentSettings {
    fn eq(&self, other: &Self) -> bool {
        let AgentSettings {
            mechanics,
            interaction,
            growth_rate,
            growth_rate_setter,
            spring_length_threshold,
            spring_length_threshold_setter,
            neighbor_reduction,
        } = &self;
        Python::attach(|py| {
            use core::ops::Deref;
            mechanics.borrow(py).deref().eq(&other.mechanics.borrow(py))
                && interaction
                    .borrow(py)
                    .deref()
                    .eq(&other.interaction.borrow(py))
                && growth_rate.eq(&other.growth_rate)
                && growth_rate_setter
                    .borrow(py)
                    .deref()
                    .eq(&other.growth_rate_setter.borrow(py))
                && spring_length_threshold.eq(&other.spring_length_threshold)
                && spring_length_threshold_setter
                    .borrow(py)
                    .deref()
                    .eq(&other.spring_length_threshold_setter.borrow(py))
                && neighbor_reduction.eq(&other.neighbor_reduction)
        })
    }
}

impl From<AgentSettings> for RodAgent {
    fn from(value: AgentSettings) -> Self {
        Python::attach(|py| {
            let AgentSettings {
                mechanics,
                interaction,
                growth_rate,
                growth_rate_setter,
                spring_length_threshold,
                spring_length_threshold_setter,
                neighbor_reduction,
            } = value;
            let mechanics = mechanics.borrow(py).clone().into();
            let interaction = RodInteraction(interaction.borrow(py).clone());
            let growth_rate_setter = growth_rate_setter.borrow(py).clone();
            let spring_length_threshold_setter = spring_length_threshold_setter.borrow(py).clone();
            RodAgent {
                mechanics,
                interaction,
                growth_rate,
                growth_rate_setter,
                spring_length_threshold,
                spring_length_threshold_setter,
                neighbor_reduction,
            }
        })
    }
}

#[pymethods]
impl AgentSettings {
    /// Constructs a new :class:`AgentSettings` class.
    ///
    /// Similarly to the :class:`Configuration` class, this constructor takes `**kwargs` and sets
    /// attributes accordingly.
    /// If a given attribute is not present in the base of :class:`AgentSettings` it will be
    /// passed on to
    /// :class:`RodMechanicsSettings` and :class:`MorsePotentialF32`.
    #[new]
    #[pyo3(signature = (**kwds))]
    pub fn new(py: Python, kwds: Option<&Bound<pyo3::types::PyDict>>) -> pyo3::PyResult<Py<Self>> {
        let as_new = Py::new(
            py,
            AgentSettings {
                mechanics: Py::new(py, RodMechanicsSettings::default())?,
                interaction: Py::new(
                    py,
                    PhysicalInteraction(
                        PhysInt::MorsePotentialF32(cellular_raza::prelude::MorsePotentialF32 {
                            radius: 3.0,              // MICROMETRE
                            potential_stiffness: 0.5, // 1/MICROMETRE
                            cutoff: 10.0,             // MICROMETRE
                            strength: 0.1,            // MICROMETRE^2 / MIN^2
                        }),
                        0,
                    ),
                )?,
                growth_rate: 0.01,
                growth_rate_setter: Py::new(
                    py,
                    GrowthRateSetter::NormalDistr {
                        mean: 0.01,
                        std: 0.,
                    },
                )?,
                spring_length_threshold: 6.0,
                spring_length_threshold_setter: Py::new(
                    py,
                    SpringLengthThresholdSetter::Explicit { l1: 6.0, l2: 6.0 },
                )?,
                neighbor_reduction: None,
            },
        )?;
        if let Some(kwds) = kwds {
            for (key, value) in kwds.iter() {
                let key: Py<PyString> = key.extract()?;
                match as_new.getattr(py, &key) {
                    Ok(_) => {
                        if key.to_str(py)? == "growth_rate_setter" {
                            let grs = GrowthRateSetter::from_pydict(&value.cast_into()?)?;
                            as_new.borrow_mut(py).growth_rate_setter = Py::new(py, grs)?;
                        } else {
                            as_new.setattr(py, &key, value)?
                        }
                    }
                    Err(e) => {
                        let as_new = as_new.borrow_mut(py);
                        match (
                            as_new.interaction.getattr(py, &key),
                            as_new.mechanics.getattr(py, &key),
                        ) {
                            (Ok(_), _) => as_new.interaction.setattr(py, &key, value)?,
                            (Err(_), Ok(_)) => as_new.mechanics.setattr(py, &key, value)?,
                            (Err(_), Err(_)) => Err(e)?,
                        }
                    }
                }
            }
        }
        Ok(as_new)
    }

    /// Formats and prints the :class:`AgentSettings`
    pub fn __repr__(&self, py: Python) -> PyResult<String> {
        use std::io::Write;
        let mut out = Vec::new();
        writeln!(out, "AgentSettings {{")?;
        writeln!(out, "{}", self.mechanics.call_method0(py, "__repr__")?,)?;
        writeln!(out, "{}", self.interaction.call_method0(py, "__repr__")?)?;
        writeln!(out, "growth_rate: {}", self.growth_rate)?;
        writeln!(
            out,
            "spring_length_threshold: {}",
            self.spring_length_threshold
        )?;
        writeln!(out, "}}")?;
        Ok(String::from_utf8(out)?)
    }

    /// Converts the class to a dictionary
    pub fn to_rod_agent_dict<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let Self {
            mechanics,
            interaction,
            growth_rate,
            growth_rate_setter,
            spring_length_threshold,
            spring_length_threshold_setter,
            neighbor_reduction,
        } = self;
        use pyo3::types::IntoPyDict;
        let res = [
            (
                "diffusion_constant",
                mechanics.getattr(py, "diffusion_constant")?,
            ),
            ("spring_tension", mechanics.getattr(py, "spring_tension")?),
            ("rigidity", mechanics.getattr(py, "rigidity")?),
            ("spring_length", mechanics.getattr(py, "spring_length")?),
            (
                "spring_length_threshold_setter",
                spring_length_threshold_setter
                    .borrow(py)
                    .to_pydict(py)?
                    .into_any()
                    .unbind(),
            ),
            ("damping", mechanics.getattr(py, "damping")?),
            (
                "interaction",
                interaction
                    .clone()
                    .into_pyobject_or_pyerr(py)?
                    .into_any()
                    .unbind(),
            ),
            (
                "growth_rate_setter",
                growth_rate_setter
                    .borrow(py)
                    .to_pydict(py)?
                    .into_any()
                    .unbind(),
            ),
            (
                "growth_rate",
                pyo3::types::PyFloat::new(py, *growth_rate as f64)
                    .into_any()
                    .unbind(),
            ),
            (
                "spring_length_threshold",
                pyo3::types::PyFloat::new(py, *spring_length_threshold as f64)
                    .into_any()
                    .unbind(),
            ),
            (
                "neighbor_reduction",
                neighbor_reduction.into_pyobject(py)?.into_any().unbind(),
            ),
        ]
        .into_py_dict(py)?;
        Ok(res)
    }
}

/// Contains all settings needed to configure the simulation
#[pyclass(set_all, get_all, module = "cr_mech_coli")]
#[derive(Clone, Debug, Deserialize, Serialize, AbsDiffEq, PartialEq)]
#[approx(epsilon_type = f32)]
pub struct Configuration {
    /// Number of threads used for solving the system.
    #[approx(equal)]
    pub n_threads: NonZeroUsize,
    /// Starting time
    pub t0: f32,
    /// Time increment
    pub dt: f32,
    /// Maximum solving time
    pub t_max: f32,
    /// Interval in which results will be saved
    #[approx(equal)]
    pub n_saves: usize,
    /// Specifies if a progress bar should be shown during the solving process.
    #[approx(skip)]
    pub progressbar: Option<String>,
    /// Overall domain size of the simulation. This may determine an upper bound on the number of
    /// agents which can be put into the simulation.
    #[approx(into_iter)]
    pub domain_size: [f32; 2],
    /// We assume that the domain is a thin 3D slice. This specifies the height of the domain.
    pub domain_height: f32,
    /// Number of voxels used to solve the system. This may yield performance improvements but
    /// specifying a too high number will yield incorrect results.
    /// See also https://cellular-raza.com/internals/concepts/domain/decomposition/.
    #[approx(equal)]
    pub n_voxels: [usize; 2],
    /// Initial seed for randomizations. This can be useful to run multiple simulations with
    /// identical parameters but slightly varying initial conditions.
    #[approx(equal)]
    pub rng_seed: u64,
    /// See [cellular_raza-building_blocks::CartesianSubDomainRods]
    pub gel_pressure: f32,
    /// See [cellular_raza-building_blocks::CartesianCuboidRods]
    pub surface_friction: f32,
    /// See [cellular_raza-building_blocks::CartesianCuboidRods]
    pub surface_friction_distance: f32,
    /// Define in which format to store results. Only uses memory by default
    #[approx(skip)]
    pub storage_options: Vec<StorageOption>,
    /// Store results in this given path. Only takes effect if more StorageOptions than memory are
    /// utilized.
    #[approx(skip)]
    pub storage_location: std::path::PathBuf,
    /// Appends a suffix to the date which is generated automatically.
    #[approx(skip)]
    pub storage_suffix: Option<std::path::PathBuf>,
}

impl Default for Configuration {
    fn default() -> Self {
        Self {
            n_threads: 1.try_into().unwrap(),
            t0: 0.0,      // MIN
            dt: 0.1,      // MIN
            t_max: 100.0, // MIN
            n_saves: 10,  // N_Samples
            progressbar: None,
            domain_size: [100.0; 2], // MICROMETRE
            domain_height: 2.5,      // MICROMETRE
            n_voxels: [1; 2],
            rng_seed: 0,
            gel_pressure: 0.,
            surface_friction: 0.,
            surface_friction_distance: 1.,
            storage_options: vec![StorageOption::Memory],
            storage_location: std::path::PathBuf::from("out"),
            storage_suffix: None,
        }
    }
}

#[pymethods]
impl Configuration {
    /// Constructs a new :class:`Configuration` class
    ///
    /// The constructor `Configuration(**kwargs)` takes a dictionary as an optional argument.
    /// This allows to easily set variables in a pythoic manner.
    /// In addition, every argument which is not an attribute of :class:`Configuration` will be
    /// passed onwards to the :class:`AgentSettings` field.
    #[new]
    #[pyo3(signature = (**kwds))]
    pub fn new(py: Python, kwds: Option<&Bound<pyo3::types::PyDict>>) -> pyo3::PyResult<Py<Self>> {
        let res_new = Py::new(py, Self::default())?;
        if let Some(kwds) = kwds {
            for (key, value) in kwds.iter() {
                let key: Py<PyString> = key.extract()?;
                res_new.setattr(py, &key, value)?;
            }
        }
        Ok(res_new)
    }

    /// Returns an identical clone of the current object
    pub fn __deepcopy__(&self, _memo: pyo3::Bound<pyo3::types::PyDict>) -> Self {
        self.clone()
    }

    /// Formats and prints the :class:`Configuration`
    pub fn __repr__(&self) -> String {
        format!("{:#?}", self)
    }

    /// Serializes this struct to the json format
    pub fn to_json(&self) -> PyResult<String> {
        let res = serde_json::to_string_pretty(&self);
        res.map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e}")))
    }

    /// Deserializes this struct from a json string
    #[staticmethod]
    pub fn from_json(json_string: Bound<PyString>) -> PyResult<Self> {
        let json_str = json_string.to_str()?;
        let res = serde_json::from_str(json_str);
        res.map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e}")))
    }

    /// Attempts to create a hash from the contents of this :class:`Configuration`.
    /// Warning: This feature is experimental.
    pub fn to_hash(&self) -> PyResult<u64> {
        let json_string = self.to_json()?;
        let mut hasher = std::hash::DefaultHasher::new();
        hasher.write(json_string.as_bytes());
        Ok(hasher.finish())
    }

    /// Parses the content of a given toml file and returns a :class:`Configuration` object which
    /// contains the given values.
    /// See also :func:`~Configuration.from_toml_string`.
    #[staticmethod]
    pub fn from_toml(filename: String) -> PyResult<Self> {
        let content = std::fs::read_to_string(filename)?;
        Self::from_toml_string(&content)
    }

    /// Parses the contents of a given string and returns a :class:`Configuration`.
    /// See also :func:`~Configuration.from_toml_string`.
    #[staticmethod]
    pub fn from_toml_string(toml_string: &str) -> PyResult<Self> {
        toml::from_str(toml_string)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))
    }

    /// TODO
    #[staticmethod]
    pub fn deserialize(data: Vec<u8>) -> PyResult<Self> {
        serde_pickle::from_slice(&data, Default::default())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))
    }

    /// TODO
    pub fn __reduce__<'py>(
        &'py self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        use std::ffi::CString;
        py.run(
            &CString::new("from cr_mech_coli import Configuration")?,
            None,
            None,
        )?;
        let deserialize = py.eval(&CString::new("Configuration.deserialize")?, None, None)?;
        let data = serde_pickle::to_vec(&self, Default::default())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
        Ok((
            deserialize.into_pyobject_or_pyerr(py)?.into_any(),
            (data,).into_pyobject_or_pyerr(py)?.into_any(),
        ))
    }
}

mod test_config {
    #[test]
    fn test_hashing() {
        use super::*;
        Python::initialize();
        Python::attach(|py| {
            let c1 = Configuration::new(py, None).unwrap();
            let c2 = Configuration::new(py, None).unwrap();
            c2.setattr(py, "n_saves", 100).unwrap();
            let h1 = c1.borrow(py).to_hash().unwrap();
            let h2 = c2.borrow(py).to_hash().unwrap();
            assert!(h1 != h2);
        });
    }

    #[test]
    fn test_parse_toml() {
        use super::*;
        Python::initialize();
        let toml_string = "
n_threads=1
t0=0.0
dt=0.1
t_max=100.0
n_saves=10
domain_size=[100.0, 100.0]
domain_height=2.5
n_voxels=[1, 1]
rng_seed=0
gel_pressure=0
surface_friction=0
surface_friction_distance=1
storage_options=['Memory']
storage_location='out'
"
        .to_string();
        let config: Configuration = Configuration::from_toml_string(&toml_string).unwrap();
        assert_eq!(config.dt, 0.1);
        assert_eq!(config.t_max, 100.0);
    }
}
