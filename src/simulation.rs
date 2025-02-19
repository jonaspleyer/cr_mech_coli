use std::{hash::Hasher, num::NonZeroUsize};

use backend::chili::SimulationError;
use cellular_raza::prelude::*;
use numpy::{PyArrayMethods, PyUntypedArrayMethods, ToPyArray};
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
        format!("{:?}", self)
    }

    #[getter]
    fn pos<'a>(&'a self, py: Python<'a>) -> Bound<'a, numpy::PyArray2<f32>> {
        use numpy::ToPyArray;
        let nrows = self.pos.nrows();
        let new_array =
            numpy::nalgebra::MatrixXx3::from_iterator(nrows, self.pos.iter().map(|&x| x));
        new_array.to_pyarray_bound(py)
    }

    #[setter]
    fn set_pos<'a>(&'a mut self, pos: Bound<'a, numpy::PyArray2<f32>>) -> pyo3::PyResult<()> {
        use numpy::PyArrayMethods;
        let nrows = pos.shape()[0];
        let iter: Vec<f32> = pos.to_vec()?;
        self.pos = nalgebra::MatrixXx3::<f32>::from_iterator(nrows, iter.into_iter());
        Ok(())
    }

    #[getter]
    fn vel<'a>(&'a self, py: Python<'a>) -> Bound<'a, numpy::PyArray2<f32>> {
        use numpy::ToPyArray;
        let new_array = numpy::nalgebra::MatrixXx3::<f32>::from_iterator(
            self.vel.nrows(),
            self.vel.iter().map(|&x| x),
        );
        new_array.to_pyarray_bound(py)
    }

    #[setter]
    fn set_vel<'a>(&'a mut self, pos: Bound<'a, numpy::PyArray2<f32>>) -> pyo3::PyResult<()> {
        use numpy::PyArrayMethods;
        let nrows = pos.shape()[0];
        let iter: Vec<f32> = pos.to_vec()?;
        self.vel = nalgebra::MatrixXx3::<f32>::from_iterator(nrows, iter.into_iter());
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
#[pyclass(get_all, set_all)]
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
                    PhysicalInteraction::MorsePotentialF32(MorsePotentialF32 {
                        radius: 3.0,              // MICROMETRE
                        potential_stiffness: 0.5, // 1/MICROMETRE
                        cutoff: 10.0,             // MICROMETRE
                        strength: 0.1,            // MICROMETRE^2 / MIN^2
                    }),
                )?,
                growth_rate: 0.1,
                spring_length_threshold: 6.0,
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
    pub fn __repr__(&self) -> String {
        format!("{:#?}", self)
    }
}

/// Contains all settings needed to configure the simulation
#[pyclass(set_all, get_all)]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Configuration {
    /// Number of agents to put into the simulation. Depending on the size specified, this number
    /// may be lowered artificially to account for the required space.
    pub n_agents: usize,
    /// Number of threads used for solving the system.
    pub n_threads: NonZeroUsize,
    /// Starting time
    pub t0: f32,
    /// Time increment
    pub dt: f32,
    /// Maximum solving time
    pub t_max: f32,
    /// Interval in which results will be saved
    pub save_interval: f32,
    /// Specifies if a progress bar should be shown during the solving process.
    pub show_progressbar: bool,
    /// Overall domain size of the simulation. This may determine an upper bound on the number of
    /// agents which can be put into the simulation.
    pub domain_size: f32,
    /// We assume that the domain is a thin 3D slice. This specifies the height of the domain.
    pub domain_height: f32,
    /// Determines the amount with which positions should be randomized. Should be a value between
    /// `0.0` and `1.0`.
    pub randomize_position: f32,
    /// Number of voxels used to solve the system. This may yield performance improvements but
    /// specifying a too high number will yield incorrect results.
    /// See also https://cellular-raza.com/internals/concepts/domain/decomposition/.
    pub n_voxels: usize,
    /// Initial seed for randomizations. This can be useful to run multiple simulations with
    /// identical parameters but slightly varying initial conditions.
    pub rng_seed: u64,
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
        let res_new = Py::new(
            py,
            Self {
                n_agents: 2,
                n_threads: 1.try_into().unwrap(),
                t0: 0.0,             // MIN
                dt: 0.1,             // MIN
                t_max: 100.0,        // MIN
                save_interval: 10.0, // MIN
                show_progressbar: false,
                domain_size: 100.0, // MICROMETRE
                domain_height: 2.5, // MICROMETRE
                randomize_position: 0.01,
                n_voxels: 1,
                rng_seed: 0,
            },
        )?;
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
    /// This will insert default values of not specified otherwise.
    #[staticmethod]
    pub fn from_toml(py: Python, toml_string: String) -> PyResult<Py<Self>> {
        // let out = Self::new(py, None)?;
        let out: Self = toml::from_str(&toml_string)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
        /* for (key, value) in table.iter() {
            use toml::Value::*;
            match value {
                String(string) => out.setattr(py, PyString::new_bound(py, key), string)?,
                Integer(int) => out.setattr(py, PyString::new_bound(py, key), *int)?,
                Float(float) => out.setattr(py, PyString::new_bound(py, key), *float)?,
                Boolean(boolean) => out.setattr(py, PyString::new_bound(py, key), *boolean)?,
                Datetime(_) | Array(_) | Table(_) => unimplemented!(),
            }
        }*/
        Py::new(py, out)
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
            c2.setattr(py, "save_interval", 100.0).unwrap();
            let h1 = c1.borrow(py).to_hash().unwrap();
            let h2 = c2.borrow(py).to_hash().unwrap();
            assert!(h1 != h2);
        });
    }

    #[test]
    fn test_parse_toml() {
        use super::*;
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let toml_string = "
n_agents=2
n_threads=1
t0=0.0
dt=0.1
t_max=100.0
save_interval=10.0
show_progressbar=false
domain_size=100.0
domain_height=2.5
randomize_position=0.01
n_voxels=1
rng_seed=0
"
            .to_string();
            let config: Configuration = Configuration::from_toml(py, toml_string)
                .unwrap()
                .extract(py)
                .unwrap();
            let toml_string = toml::to_string(&config).unwrap();
            println!("{toml_string}");
            assert_eq!(config.dt, 0.1);
            assert_eq!(config.t_max, 100.0);
        })
    }
}

prepare_types!(
    aspects: [Mechanics, Interaction, Cycle],
);

/// Creates positions for multiple :class`RodAgent`s which can be used for simulation purposes.
#[pyfunction]
#[pyo3(signature = (
    n_agents,
    agent_settings,
    config,
    rng_seed = 0,
    dx = 0.0,
    randomize_positions = 0.0,
    n_vertices = 8,
))]
pub fn generate_positions_old<'py>(
    py: Python<'py>,
    n_agents: usize,
    agent_settings: &AgentSettings,
    config: &Configuration,
    rng_seed: u64,
    dx: f32,
    randomize_positions: f32,
    n_vertices: usize,
) -> PyResult<Vec<Bound<'py, numpy::PyArray2<f32>>>> {
    // numpy::nalgebra::DMatrix<f32>
    use rand::seq::IteratorRandom;
    use rand::Rng;
    use rand_chacha::rand_core::SeedableRng;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(rng_seed);
    let mechanics: RodMechanicsSettings = agent_settings.mechanics.extract(py)?;
    let spring_length = mechanics.spring_length;
    let s = randomize_positions.clamp(0.0, 1.0);

    // Split the domain into chunks
    let n_chunk_sides = (n_agents as f32).sqrt().ceil() as usize;
    let dchunk = (config.domain_size - 2.0 * dx) / n_chunk_sides as f32;
    let all_indices = itertools::iproduct!(0..n_chunk_sides, 0..n_chunk_sides);
    let picked_indices = all_indices.choose_multiple(&mut rng, n_agents);
    let drod_length_half = (n_vertices as f32) * spring_length / 2.0;

    Ok(picked_indices
        .into_iter()
        .map(|index| {
            let xlow = dx + index.0 as f32 * dchunk;
            let ylow = dx + index.1 as f32 * dchunk;
            let middle = numpy::array![
                rng.gen_range(xlow + drod_length_half..xlow + dchunk - drod_length_half),
                rng.gen_range(ylow + drod_length_half..ylow + dchunk - drod_length_half),
                rng.gen_range(0.4 * config.domain_height..0.6 * config.domain_height),
            ];
            let angle: f32 = rng.gen_range(0.0..2.0 * std::f32::consts::PI);
            let p1 = middle - drod_length_half * numpy::array![angle.cos(), angle.sin(), 0.0];
            fn s_gen(x: f32, rng: &mut rand_chacha::ChaCha8Rng) -> f32 {
                if x == 0.0 {
                    1.0
                } else {
                    rng.gen_range(1.0 - x..1.0 + x)
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
            .to_pyarray_bound(py)
        })
        .collect())
}

#[test]
fn backwards_compat_generate_positions_old() -> PyResult<()> {
    pyo3::prepare_freethreaded_python();
    let generated_pos = Python::with_gil(|py| {
        let agent_settings = AgentSettings::new(py, None)?.extract(py)?;
        let config = Configuration::new(py, None)?.extract(py)?;
        let old_positions =
            generate_positions_old(py, 4, &agent_settings, &config, 1, 0.0, 0.1, 8)?;
        PyResult::Ok(
            old_positions
                .into_iter()
                .map(|pos| pos.to_owned_array())
                .collect::<Vec<_>>(),
        )
    })?;
    let old_pos = vec![
        numpy::array![
            [27.468908, 20.12428, 1.4922986],
            [30.074106, 21.726332, 1.4922986],
            [33.135212, 22.993032, 1.4922986],
            [35.8945, 24.686779, 1.4922986],
            [37.985126, 25.82119, 1.4922986],
            [41.40438, 27.021421, 1.4922986],
            [41.46813, 28.72572, 1.4922986],
            [44.44608, 31.503601, 1.4922986]
        ],
        numpy::array![
            [8.114415, 69.866714, 1.4667264],
            [10.9589405, 70.1071, 1.4667264],
            [14.553812, 70.36222, 1.4667264],
            [17.65355, 70.67615, 1.4667264],
            [19.406654, 70.78856, 1.4667264],
            [21.934778, 71.24877, 1.4667264],
            [24.32788, 71.43905, 1.4667264],
            [27.831997, 71.5903, 1.4667264]
        ],
        numpy::array![
            [83.25456, 23.841858, 1.384213],
            [86.2094, 23.719553, 1.384213],
            [87.3836, 22.407425, 1.384213],
            [89.15912, 19.916075, 1.384213],
            [94.61067, 19.968637, 1.384213],
            [98.563545, 18.060032, 1.384213],
            [97.98949, 17.147717, 1.384213],
            [104.652954, 12.351942, 1.384213]
        ],
        numpy::array![
            [86.79208, 71.02558, 1.4164526],
            [85.146545, 73.689835, 1.4164526],
            [82.56933, 74.24946, 1.4164526],
            [81.98949, 77.84386, 1.4164526],
            [78.248146, 77.59455, 1.4164526],
            [76.69923, 82.323654, 1.4164526],
            [75.966194, 86.90044, 1.4164526],
            [72.31987, 89.531456, 1.4164526]
        ],
    ];
    for (p, q) in generated_pos.into_iter().zip(old_pos.into_iter()) {
        assert_eq!(p, q);
    }
    Ok(())
}

/// Executes a simulation given a :class:`Configuration` and a list of :class:`RodAgent`.
#[pyfunction]
pub fn run_simulation_with_agents(
    config: &Configuration,
    agents: Vec<RodAgent>,
) -> pyo3::PyResult<CellContainer> {
    // TODO after initializing this state, we need to check that it is actually valid
    let t0 = config.t0;
    let dt = config.dt;
    let t_max = config.t_max;
    let save_interval = config.save_interval;
    let time = FixedStepsize::from_partial_save_interval(t0, dt, t_max, save_interval)
        .map_err(SimulationError::from)?;
    let storage = StorageBuilder::new().priority([StorageOption::Memory]);
    let settings = Settings {
        n_threads: config.n_threads,
        time,
        storage,
        show_progressbar: config.show_progressbar,
    };

    let mut domain = CartesianCuboid::from_boundaries_and_n_voxels(
        [0.0; 3],
        [config.domain_size, config.domain_size, config.domain_height],
        [config.n_voxels, config.n_voxels, 1],
    )
    .map_err(SimulationError::from)?;
    domain.rng_seed = config.rng_seed;
    let domain = CartesianCuboidRods { domain };

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
        aspects: [Mechanics, Interaction, Cycle],
        zero_force_default: |c: &RodAgent| {
            nalgebra::MatrixXx3::zeros(c.mechanics.pos().nrows())
        },
    )?;
    let cells = storage
        .cells
        .load_all_elements()
        .unwrap()
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

    CellContainer::new(cells)
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
