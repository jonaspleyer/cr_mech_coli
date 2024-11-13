use std::{hash::Hasher, num::NonZeroUsize};

use backend::chili::SimulationError;
use cellular_raza::prelude::*;
use numpy::PyUntypedArrayMethods;
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
            rigidity: 0.5,
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
    pub interaction: Py<MorsePotentialF32>,
    /// Rate with which the length of the bacterium grows
    pub growth_rate: f32,
    /// Threshold when the bacterium divides
    pub spring_length_threshold: f32,
    /// Number of vertices to use for this agent
    pub n_vertices: usize,
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
                    MorsePotentialF32 {
                        radius: 3.0,              // MICROMETRE
                        potential_stiffness: 0.5, // 1/MICROMETRE
                        cutoff: 10.0,             // MICROMETRE
                        strength: 0.1,            // MICROMETRE^2 / MIN^2
                    },
                )?,
                growth_rate: 0.1,
                spring_length_threshold: 6.0,
                n_vertices: 8,
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
    /// Contains a template for defining multiple :class:`RodAgent` of the simulation.
    pub agent_settings: Py<AgentSettings>,
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
                agent_settings: Py::new(py, AgentSettings::new(py, None)?)?,
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
                match res_new.getattr(py, &key) {
                    Ok(_) => res_new.setattr(py, &key, value)?,
                    Err(_) => res_new
                        .borrow_mut(py)
                        .agent_settings
                        .setattr(py, &key, value)?,
                }
            }
        }
        Ok(res_new)
    }

    /// Formats and prints the :class:`Configuration`
    pub fn __repr__(&self) -> String {
        format!("{:#?}", self)
    }

    /// Serializes this struct to the json format
    pub fn to_json(&self) -> PyResult<String> {
        let res = serde_json::to_string_pretty(&self);
        Ok(res.or_else(|e| Err(pyo3::exceptions::PyIOError::new_err(format!("{e}"))))?)
    }

    /// Deserializes this struct from a json string
    #[staticmethod]
    pub fn from_json(json_string: Bound<PyString>) -> PyResult<Self> {
        let json_str = json_string.to_str()?;
        let res = serde_json::from_str(&json_str);
        Ok(res.or_else(|e| Err(pyo3::exceptions::PyIOError::new_err(format!("{e}"))))?)
    }

    /// Attempts to create a hash from the contents of this :class:`Configuration`.
    /// Warning: This feature is experimental.
    pub fn to_hash(&self) -> PyResult<u64> {
        let json_string = self.to_json()?;
        let mut hasher = std::hash::DefaultHasher::new();
        hasher.write(&json_string.as_bytes());
        Ok(hasher.finish())
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
            c2.borrow_mut(py)
                .agent_settings
                .setattr(py, "growth_rate", 200.0)
                .unwrap();
            let h1 = c1.borrow(py).to_hash().unwrap();
            let h2 = c2.borrow(py).to_hash().unwrap();
            assert!(h1 != h2);
        });
    }
}

prepare_types!(
    aspects: [Mechanics, Interaction, Cycle],
);

/// Executes the simulation with the given :class:`Configuration`
#[pyfunction]
pub fn run_simulation(config: Configuration) -> pyo3::PyResult<CellContainer> {
    use rand::Rng;
    use rand_chacha::rand_core::SeedableRng;
    Python::with_gil(|py| {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(config.rng_seed);
        let agent_settings: AgentSettings = config.agent_settings.extract(py)?;
        let mechanics: RodMechanicsSettings = agent_settings.mechanics.extract(py)?;
        let interaction: MorsePotentialF32 = agent_settings.interaction.extract(py)?;
        let spring_length = mechanics.spring_length;
        let dx = spring_length * mechanics.pos.nrows() as f32;
        let s = config.randomize_position;
        let bacteria = (0..config.n_agents).map(|_| {
            // TODO make these positions much more spaced
            let p1 = [
                rng.gen_range(dx..config.domain_size - dx),
                rng.gen_range(dx..config.domain_size - dx),
                rng.gen_range(0.4 * config.domain_height..0.6 * config.domain_height),
            ];
            let angle: f32 = rng.gen_range(0.0..2.0 * std::f32::consts::PI);
            RodAgent {
                mechanics: RodMechanics {
                    pos: nalgebra::MatrixXx3::<f32>::from_fn(agent_settings.n_vertices, |r, c| {
                        p1[c]
                            + r as f32
                                * spring_length
                                * rng.gen_range(1.0 - s..1.0 + s)
                                * if c == 0 {
                                    (angle * rng.gen_range(1.0 - s..1.0 + s)).cos()
                                } else if c == 1 {
                                    (angle * rng.gen_range(1.0 - s..1.0 + s)).sin()
                                } else {
                                    0.0
                                }
                    }),
                    vel: nalgebra::MatrixXx3::<f32>::from_fn(agent_settings.n_vertices, |_, _| 0.0),
                    diffusion_constant: mechanics.diffusion_constant,
                    spring_tension: mechanics.spring_tension,
                    rigidity: mechanics.rigidity,
                    spring_length: mechanics.spring_length,
                    damping: mechanics.damping,
                },
                interaction: RodInteraction(interaction.clone()),
                growth_rate: agent_settings.growth_rate,
                spring_length_threshold: agent_settings.spring_length_threshold,
            }
        });

        // TODO after initializing this state, we need to check that it is actually valid

        let t0 = config.t0;
        let dt = config.dt;
        let t_max = config.t_max;
        let save_interval = config.save_interval;
        let time = FixedStepsize::from_partial_save_interval(t0, dt, t_max, save_interval)
            .or_else(|x| Err(SimulationError::from(x)))?;
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
        .or_else(|x| Err(SimulationError::from(x)))?;
        domain.rng_seed = config.rng_seed;
        let domain = CartesianCuboidRods { domain };

        test_compatibility!(
            aspects: [Mechanics, Interaction, Cycle],
            domain: domain,
            agents: bacteria,
            settings: settings,
        );
        let storage = run_main!(
            agents: bacteria,
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
                        .map(|(ident, (cbox, _))| (ident, (cbox.cell.into_py(py), cbox.parent)))
                        .collect(),
                )
            })
            .collect();

        Ok(CellContainer::new(cells)?)
    })
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
