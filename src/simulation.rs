use std::{hash::Hasher, num::NonZeroUsize};

use backend::chili::SimulationError;
use cellular_raza::prelude::*;
use numpy::{PyUntypedArrayMethods, ToPyArray};
use pyo3::IntoPyObjectExt;
use pyo3::{prelude::*, types::PyString};
use serde::{Deserialize, Serialize};
use time::FixedStepsize;

use crate::datatypes::CellContainer;

use crate::agent::*;

/// Contains all settings required to construct :class:`RodMechanics`
#[pyclass]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RodMechanicsSettings {
    /// The current position
    pub pos: nalgebra::MatrixXx3<f32>,
    /// The current velocity
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
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AgentSettings {
    /// Settings for the mechanics part of :class:`RodAgent`. See also :class:`RodMechanicsSettings`.
    pub mechanics: Py<RodMechanicsSettings>,
    /// Settings for the interaction part of :class:`RodAgent`. See also :class:`MorsePotentialF32`.
    pub interaction: Py<PhysicalInteraction>,
    /// Rate with which the length of the bacterium grows
    pub growth_rate: f32,
    /// Threshold when the bacterium divides
    pub spring_length_threshold: f32,
    /// Reduces the growth rate with multiplier $((max - N)/max)^q $
    pub neighbor_reduction: Option<(usize, f32)>,
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
                        PhysInt::MorsePotentialF32(MorsePotentialF32 {
                            radius: 3.0,              // MICROMETRE
                            potential_stiffness: 0.5, // 1/MICROMETRE
                            cutoff: 10.0,             // MICROMETRE
                            strength: 0.1,            // MICROMETRE^2 / MIN^2
                        }),
                        0,
                    ),
                )?,
                growth_rate: 0.1,
                spring_length_threshold: 6.0,
                neighbor_reduction: None,
            },
        )?;
        if let Some(kwds) = kwds {
            for (key, value) in kwds.iter() {
                let key: Py<PyString> = key.extract()?;
                match as_new.getattr(py, &key) {
                    Ok(_) => as_new.setattr(py, &key, value)?,
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
            spring_length_threshold,
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
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Configuration {
    /// Number of threads used for solving the system.
    pub n_threads: NonZeroUsize,
    /// Starting time
    pub t0: f32,
    /// Time increment
    pub dt: f32,
    /// Maximum solving time
    pub t_max: f32,
    /// Interval in which results will be saved
    pub n_saves: usize,
    /// Specifies if a progress bar should be shown during the solving process.
    pub show_progressbar: bool,
    /// Overall domain size of the simulation. This may determine an upper bound on the number of
    /// agents which can be put into the simulation.
    pub domain_size: [f32; 2],
    /// We assume that the domain is a thin 3D slice. This specifies the height of the domain.
    pub domain_height: f32,
    /// Number of voxels used to solve the system. This may yield performance improvements but
    /// specifying a too high number will yield incorrect results.
    /// See also https://cellular-raza.com/internals/concepts/domain/decomposition/.
    pub n_voxels: [usize; 2],
    /// Initial seed for randomizations. This can be useful to run multiple simulations with
    /// identical parameters but slightly varying initial conditions.
    pub rng_seed: u64,
    /// See [cellular_raza-building_blocks::CartesianSubDomainRods]
    pub gravity: f32,
    /// See [cellular_raza-building_blocks::CartesianCuboidRods]
    pub surface_friction: f32,
    /// See [cellular_raza-building_blocks::CartesianCuboidRods]
    pub surface_friction_distance: f32,
    /// Define in which format to store results. Only uses memory by default
    pub storage_options: Vec<StorageOption>,
    /// Store results in this given path. Only takes effect if more StorageOptions than memory are
    /// utilized.
    pub storage_location: std::path::PathBuf,
}

impl Default for Configuration {
    fn default() -> Self {
        Self {
            n_threads: 1.try_into().unwrap(),
            t0: 0.0,      // MIN
            dt: 0.1,      // MIN
            t_max: 100.0, // MIN
            n_saves: 10,  // N_Samples
            show_progressbar: false,
            domain_size: [100.0; 2], // MICROMETRE
            domain_height: 2.5,      // MICROMETRE
            n_voxels: [1; 2],
            rng_seed: 0,
            gravity: 0.,
            surface_friction: 0.,
            surface_friction_distance: 1.,
            storage_options: vec![StorageOption::Memory],
            storage_location: std::path::PathBuf::from("out"),
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
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
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
        pyo3::prepare_freethreaded_python();
        let toml_string = "
n_threads=1
t0=0.0
dt=0.1
t_max=100.0
n_saves=10
show_progressbar=false
domain_size=[100.0, 100.0]
domain_height=2.5
n_voxels=[1, 1]
rng_seed=0
gravity=0
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

prepare_types!(
    aspects: [Mechanics, Interaction, Cycle],
);

/// Creates positions for multiple :class:`RodAgent` which can be used for simulation purposes.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    n_agents,
    agent_settings,
    config,
    rng_seed = 0,
    dx = [0.0, 0.0],
    randomize_positions = 0.0,
    n_vertices = 8,
))]
pub fn generate_positions_old<'py>(
    py: Python<'py>,
    n_agents: usize,
    agent_settings: &AgentSettings,
    config: &Configuration,
    rng_seed: u64,
    dx: [f32; 2],
    randomize_positions: f32,
    n_vertices: usize,
) -> PyResult<Vec<Bound<'py, numpy::PyArray2<f32>>>> {
    let mechanics: RodMechanicsSettings = agent_settings.mechanics.extract(py)?;
    Ok(_generate_positions_old(
        n_agents,
        &mechanics,
        config,
        rng_seed,
        dx,
        randomize_positions,
        n_vertices,
    )
    .into_iter()
    .map(|x| x.to_pyarray(py))
    .collect())
}

/// Backend functionality to use within rust-specific code for [generate_positions_old]
fn _generate_positions_old(
    n_agents: usize,
    mechanics: &RodMechanicsSettings,
    config: &Configuration,
    rng_seed: u64,
    dx: [f32; 2],
    randomize_positions: f32,
    n_vertices: usize,
) -> Vec<numpy::nalgebra::DMatrix<f32>> {
    // numpy::nalgebra::DMatrix<f32>
    use rand::seq::IteratorRandom;
    use rand::Rng;
    use rand_chacha::rand_core::SeedableRng;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(rng_seed);
    let spring_length = mechanics.spring_length;
    let s = randomize_positions.clamp(0.0, 1.0);

    // Split the domain into chunks
    let n_chunk_sides = (n_agents as f32).sqrt().ceil() as usize;
    let dchunk1 = (config.domain_size[0] - 2.0 * dx[0]) / n_chunk_sides as f32;
    let dchunk2 = (config.domain_size[1] - 2.0 * dx[1]) / n_chunk_sides as f32;
    let all_indices = itertools::iproduct!(0..n_chunk_sides, 0..n_chunk_sides);
    let picked_indices = all_indices.choose_multiple(&mut rng, n_agents);
    let drod_length_half = (n_vertices as f32) * spring_length / 2.0;

    picked_indices
        .into_iter()
        .map(|index| {
            let xlow = dx[0] + index.0 as f32 * dchunk1;
            let ylow = dx[1] + index.1 as f32 * dchunk2;
            let middle = numpy::array![
                rng.random_range(xlow + drod_length_half..xlow + dchunk1 - drod_length_half),
                rng.random_range(ylow + drod_length_half..ylow + dchunk2 - drod_length_half),
                rng.random_range(0.4 * config.domain_height..0.6 * config.domain_height),
            ];
            let angle: f32 = rng.random_range(0.0..2.0 * std::f32::consts::PI);
            let p1 = middle - drod_length_half * numpy::array![angle.cos(), angle.sin(), 0.0];
            fn s_gen(x: f32, rng: &mut rand_chacha::ChaCha8Rng) -> f32 {
                if x == 0.0 {
                    1.0
                } else {
                    rng.random_range(1.0 - x..1.0 + x)
                }
            }
            numpy::nalgebra::DMatrix::<f32>::from_fn(n_vertices, 3, |r, c| {
                p1[c]
                    + r as f32
                        * spring_length
                        * s_gen(s, &mut rng)
                        * if c == 0 {
                            (angle * s_gen(s, &mut rng)).cos()
                        } else if c == 1 {
                            (angle * s_gen(s, &mut rng)).sin()
                        } else {
                            0.0
                        }
            })
        })
        .collect()
}

#[test]
fn backwards_compat_generate_positions_old() -> PyResult<()> {
    let mechanics = RodMechanicsSettings::default();
    let config = Configuration::default();
    let generated_pos = _generate_positions_old(4, &mechanics, &config, 1, [0.0; 2], 0.1, 8);
    let old_pos = vec![
        numpy::nalgebra::dmatrix![
            15.782119,  16.658249,  1.4922986;
            18.387316,  18.2603,    1.4922986;
            21.448421,  19.527,     1.4922986;
            24.20771,   21.220747,  1.4922986;
            26.298336,  22.355158,  1.4922986;
            29.717592,  23.55539,   1.4922986;
            29.781338,  25.25969,   1.4922986;
            32.75929,   28.03757,   1.4922986;
        ],
        numpy::nalgebra::dmatrix![
             4.2639103, 71.299194,  1.4667264;
             7.1084356, 71.53958,   1.4667264;
            10.703307,  71.7947,    1.4667264;
            13.803044,  72.10863,   1.4667264;
            15.5561495, 72.22104,   1.4667264;
            18.084274,  72.68125,   1.4667264;
            20.477375,  72.87153,   1.4667264;
            23.981491,  73.02278,   1.4667264;
        ],
        numpy::nalgebra::dmatrix![
            68.69818,   30.033642,  1.384213;
            71.653015,  29.911337,  1.384213;
            72.82722,   28.599209,  1.384213;
            74.60274,   26.107859,  1.384213;
            80.05429,   26.160421,  1.384213;
            84.007164,  24.251816,  1.384213;
            83.433105,  23.3395,    1.384213;
            90.09657,   18.543726,  1.384213;
        ],
        numpy::nalgebra::dmatrix![
            89.117294,  63.976006,  1.4164526;
            87.471756,  66.64026,   1.4164526;
            84.89454,   67.19988,   1.4164526;
            84.3147,    70.79428,   1.4164526;
            80.57336,   70.544975,  1.4164526;
            79.02444,   75.27408,   1.4164526;
            78.291405,  79.85086,   1.4164526;
            74.64508,   82.48188,   1.4164526;
        ],
    ];
    for (p, q) in generated_pos.into_iter().zip(old_pos.into_iter()) {
        assert_eq!(p, q);
    }
    Ok(())
}

pub(crate) fn new_storage_builder(config: &Configuration) -> StorageBuilder {
    StorageBuilder::new()
        .priority(config.storage_options.clone())
        .location(&config.storage_location)
}

/// Executes a simulation given a :class:`Configuration` and a list of :class:`RodAgent`.
#[pyfunction]
pub fn run_simulation_with_agents(
    config: &Configuration,
    agents: Vec<RodAgent>,
) -> Result<CellContainer, cellular_raza::prelude::SimulationError> {
    // TODO after initializing this state, we need to check that it is actually valid
    let t0 = config.t0;
    let dt = config.dt;
    let t_max = config.t_max;
    let mut save_steps = vec![t0];
    if config.n_saves > 0 {
        let dtsave = (t_max - t0) / (config.n_saves + 1) as f32;
        save_steps.extend((1..config.n_saves + 1).map(|n| t0 + n as f32 * dtsave));
    }
    save_steps.push(t_max);
    let time = FixedStepsize::from_partial_save_points(t0, dt, save_steps)
        .map_err(SimulationError::from)?;
    let storage = new_storage_builder(config);
    let settings = Settings {
        n_threads: config.n_threads,
        time,
        storage,
        show_progressbar: config.show_progressbar,
    };

    let domain_size = [
        config.domain_size[0],
        config.domain_size[1],
        config.domain_height,
    ];
    let mut domain = CartesianCuboid::from_boundaries_and_n_voxels(
        [0.0; 3],
        domain_size,
        [config.n_voxels[0], config.n_voxels[1], 1],
    )
    .map_err(SimulationError::from)?;
    domain.rng_seed = config.rng_seed;
    let domain = CartesianCuboidRods {
        domain,
        gravity: config.gravity,
        surface_friction: config.surface_friction,
        surface_friction_distance: config.surface_friction_distance,
    };

    test_compatibility!(
        aspects: [Mechanics, Interaction, Cycle],
        domain: domain,
        agents: agents,
        settings: settings,
    );
    let storage = run_main!(
        agents: agents,
        domain: domain,
        settings: settings,
        aspects: [Mechanics, Interaction, Cycle, DomainForce],
        zero_force_default: |c: &RodAgent| {
            nalgebra::MatrixXx3::zeros(c.mechanics.pos().nrows())
        },
    )?;
    let cells = storage
        .cells
        .load_all_elements()?
        .into_iter()
        .map(|(iteration, cells)| {
            (
                iteration,
                cells
                    .into_iter()
                    .map(|(ident, (cbox, _))| (ident, (cbox.cell, cbox.parent)))
                    .collect(),
            )
        })
        .collect();

    Ok(CellContainer::new(cells).unwrap())
}

/// Sorts an iterator of :class:`CellIdentifier` deterministically.
///
/// This function is usefull for generating identical masks every simulation run.
/// This function is implemented as standalone since sorting of a :class:`CellIdentifier` is
/// typically not supported.
///
/// Args:
///     identifiers(list): A list of :class:`CellIdentifier`
///
/// Returns:
///     list: The sorted list.
#[pyfunction]
pub fn sort_cellular_identifiers(
    identifiers: Vec<CellIdentifier>,
) -> Result<Vec<CellIdentifier>, PyErr> {
    let mut identifiers = identifiers;
    identifiers.sort();
    Ok(identifiers)
}
