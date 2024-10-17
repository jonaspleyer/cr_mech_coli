use std::{hash::Hasher, num::NonZeroUsize};

use backend::chili::SimulationError;
use cellular_raza::prelude::*;
use pyo3::{prelude::*, types::PyString};
use serde::{Deserialize, Serialize};
use time::FixedStepsize;

/// Determines the number of subsections to use for each bacterial rod
pub const N_ROD_SEGMENTS: usize = 8;

/// Contains all settings required to construct [RodMechanics]
#[pyclass]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RodMechanicsSettings {
    /// The current position
    pub pos: nalgebra::SMatrix<f32, N_ROD_SEGMENTS, 3>,
    /// The current velocity
    pub vel: nalgebra::SMatrix<f32, N_ROD_SEGMENTS, 3>,
    /// Controls magnitude of32 stochastic motion
    #[pyo3(get, set)]
    pub diffusion_constant: f32,
    /// Spring tension between individual vertices
    #[pyo3(get, set)]
    pub spring_tension: f32,
    /// Stif32fness at each joint connecting two edges
    #[pyo3(get, set)]
    pub angle_stiffness: f32,
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
        let new_array = numpy::nalgebra::SMatrix::<f32, N_ROD_SEGMENTS, 3>::from_iterator(
            self.pos.iter().map(|&x| x),
        );
        new_array.to_pyarray_bound(py)
    }

    #[setter]
    fn set_pos<'a>(&'a mut self, pos: Bound<'a, numpy::PyArray2<f32>>) -> pyo3::PyResult<()> {
        use numpy::PyArrayMethods;
        let iter: Vec<f32> = pos.to_vec()?;
        self.pos = nalgebra::SMatrix::<f32, N_ROD_SEGMENTS, 3>::from_iterator(iter.into_iter());
        Ok(())
    }

    #[getter]
    fn vel<'a>(&'a self, py: Python<'a>) -> Bound<'a, numpy::PyArray2<f32>> {
        use numpy::ToPyArray;
        let new_array = numpy::nalgebra::SMatrix::<f32, N_ROD_SEGMENTS, 3>::from_iterator(
            self.vel.iter().map(|&x| x),
        );
        new_array.to_pyarray_bound(py)
    }

    #[setter]
    fn set_vel<'a>(&'a mut self, pos: Bound<'a, numpy::PyArray2<f32>>) -> pyo3::PyResult<()> {
        use numpy::PyArrayMethods;
        let iter: Vec<f32> = pos.to_vec()?;
        self.vel = nalgebra::SMatrix::<f32, N_ROD_SEGMENTS, 3>::from_iterator(iter.into_iter());
        Ok(())
    }
}

impl Default for RodMechanicsSettings {
    fn default() -> Self {
        RodMechanicsSettings {
            pos: nalgebra::SMatrix::zeros(),
            vel: nalgebra::SMatrix::zeros(),
            diffusion_constant: 0.0, // MICROMETRE^2 / MIN^2
            spring_tension: 0.1,     // 1 / MIN
            angle_stiffness: 0.05,
            spring_length: 5.0, // MICROMETRE
            damping: 0.1,       // 1/MIN
        }
    }
}

/// Contains settings needed to specify properties of the [RodAgent]
#[pyclass(get_all, set_all)]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AgentSettings {
    /// Settings for the mechanics part of [RodAgent]. See also [RodMechanicsSettings].
    pub mechanics: Py<RodMechanicsSettings>,
    /// Settings for the interaction part of [RodAgent]. See also [MorsePotentialF32].
    pub interaction: Py<MorsePotentialF32>,
    /// Rate with which the length of the bacterium grows
    pub growth_rate: f32,
    /// Threshold when the bacterium divides
    pub spring_length_threshold: f32,
}

/// Contains all settings needed to configure the simulation
#[pyclass(set_all, get_all)]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Configuration {
    agent_settings: AgentSettings,
    n_agents: usize,
    n_threads: NonZeroUsize,
    t0: f32,
    dt: f32,
    t_max: f32,
    save_interval: f32,
    show_progressbar: bool,
    domain_size: f32,
    randomize_position: f32,
    n_voxels: usize,
    rng_seed: u64,
    storage_priority: Vec<StorageOption>,
}

#[pymethods]
impl Configuration {
    /// Constructs a new [Configuration] class
    #[new]
    pub fn new(py: Python) -> pyo3::PyResult<Self> {
        Ok(Self {
            agent_settings: AgentSettings {
                mechanics: Py::new(py, RodMechanicsSettings::default())?,
                interaction: Py::new(
                    py,
                    MorsePotentialF32 {
                        radius: 5.0,                    // MICROMETRE
                        potential_stiffness: 1.0 / 5.0, // 1/MICROMETRE
                        cutoff: 8.0,                    // MICROMETRE
                        strength: 2.0,                  // MICROMETRE^2/MINUTE^2
                    },
                )?,
                growth_rate: 0.1,
                spring_length_threshold: 6.0,
            },
            n_agents: 2,
            n_threads: 1.try_into().unwrap(),
            t0: 0.0,             // MIN
            dt: 0.1,             // MIN
            t_max: 1_000.0,      // MIN
            save_interval: 50.0, // MIN
            show_progressbar: false,
            domain_size: 100.0, // MICROMETRE
            domain_height: 2.5, // MICROMETRE
            randomize_position: 0.01,
            n_voxels: 1,
            rng_seed: 0,
            storage_priority: vec![StorageOption::Memory],
        })
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

    /// Attempts to create a hash from the contents of this [Configuration].
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
            let c1 = Configuration::new(py).unwrap();
            let c2 = Configuration::new(py).unwrap();
            c2.agent_settings.setattr(py, "growth_rate", 200.0).unwrap();
            let h1 = c1.to_hash().unwrap();
            let h2 = c2.to_hash().unwrap();
            assert!(h1 != h2);
        });
    }
}

/// A basic cell-agent which makes use of
/// [RodMechanics](https://cellular-raza.com/docs/cellular_raza_building_blocks/structs.RodMechanics.html)
#[pyclass]
#[derive(CellAgent, Clone, Debug, Deserialize, Serialize)]
pub struct RodAgent {
    #[Mechanics]
    mechanics: RodMechanics<f32, N_ROD_SEGMENTS, 3>,
    #[Interaction]
    interaction: RodInteraction<MorsePotentialF32>,
    growth_rate: f32,
    spring_length_threshold: f32,
}

#[pymethods]
impl RodAgent {
    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }

    #[getter]
    fn pos<'a>(&'a self, py: Python<'a>) -> Bound<'a, numpy::PyArray2<f32>> {
        use numpy::ToPyArray;
        let new_array = numpy::nalgebra::SMatrix::<f32, N_ROD_SEGMENTS, 3>::from_iterator(
            self.mechanics.pos.iter().map(|&x| x),
        );
        new_array.to_pyarray_bound(py)
    }

    #[setter]
    fn set_pos<'a>(&'a mut self, pos: Bound<'a, numpy::PyArray2<f32>>) -> pyo3::PyResult<()> {
        use numpy::PyArrayMethods;
        let iter: Vec<f32> = pos.to_vec()?;
        self.mechanics.pos =
            nalgebra::SMatrix::<f32, N_ROD_SEGMENTS, 3>::from_iterator(iter.into_iter());
        Ok(())
    }

    #[getter]
    fn vel<'a>(&'a self, py: Python<'a>) -> Bound<'a, numpy::PyArray2<f32>> {
        use numpy::ToPyArray;
        let new_array = numpy::nalgebra::SMatrix::<f32, N_ROD_SEGMENTS, 3>::from_iterator(
            self.mechanics.vel.iter().map(|&x| x),
        );
        new_array.to_pyarray_bound(py)
    }

    #[setter]
    fn set_vel<'a>(&'a mut self, pos: Bound<'a, numpy::PyArray2<f32>>) -> pyo3::PyResult<()> {
        use numpy::PyArrayMethods;
        let iter: Vec<f32> = pos.to_vec()?;
        self.mechanics.vel =
            nalgebra::SMatrix::<f32, N_ROD_SEGMENTS, 3>::from_iterator(iter.into_iter());
        Ok(())
    }

    #[getter]
    fn radius(&self) -> f32 {
        self.interaction.0.radius
    }
}

impl Cycle<RodAgent, f32> for RodAgent {
    fn update_cycle(
            _rng: &mut rand_chacha::ChaCha8Rng,
            dt: &f32,
            cell: &mut Self,
        ) -> Option<CycleEvent> {
        cell.mechanics.spring_length += cell.growth_rate * dt;
        if cell.mechanics.spring_length > cell.spring_length_threshold {
            Some(CycleEvent::Division)
        } else {
            None
        }
    }

    fn divide(_rng: &mut rand_chacha::ChaCha8Rng, cell: &mut Self) -> Result<Self, DivisionError> {
        let c2_mechanics = cell.mechanics.divide(cell.interaction.0.radius)?;
        let mut c2 = cell.clone();
        c2.mechanics = c2_mechanics;
        Ok(c2)
    }
}

/// Resulting type when executing a full simulation
pub type SimResult = std::collections::HashMap<
    u64,
    std::collections::HashMap<CellIdentifier, (RodAgent, Option<CellIdentifier>)>,
>;

/// Executes the simulation with the given [Configuration]
#[pyfunction]
pub fn run_simulation(config: Configuration) -> Result<SimResult, PyErr> {
    use rand::Rng;
    use rand_chacha::rand_core::SeedableRng;
    Python::with_gil(|py| {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(config.rng_seed);
        let mechanics: RodMechanicsSettings = config.agent_settings.mechanics.extract(py)?;
        let interaction: MorsePotentialF32 = config.agent_settings.interaction.extract(py)?;
        let spring_length = mechanics.spring_length;
        let dx = spring_length * N_ROD_SEGMENTS as f32;
        let s = config.randomize_position;
        let bacteria = (0..config.n_agents).map(|_| {
            // TODO make these positions much more spaced
            let p1 = [
                rng.gen_range(dx..config.domain_size - dx),
                rng.gen_range(dx..config.domain_size - dx),
                rng.gen_range(2.4..2.6),
            ];
            let angle: f32 = rng.gen_range(0.0..2.0 * std::f32::consts::PI);
            RodAgent {
                mechanics: RodMechanics {
                    pos: nalgebra::SMatrix::<f32, N_ROD_SEGMENTS, 3>::from_fn(|r, c| {
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
                    vel: nalgebra::SMatrix::<f32, N_ROD_SEGMENTS, 3>::from_fn(|_, _| 0.0),
                    diffusion_constant: mechanics.diffusion_constant,
                    spring_tension: mechanics.spring_tension,
                    angle_stiffness: mechanics.angle_stiffness,
                    spring_length: mechanics.spring_length,
                    damping: mechanics.damping,
                },
                interaction: RodInteraction(interaction.clone()),
                growth_rate: config.agent_settings.growth_rate,
                spring_length_threshold: config.agent_settings.spring_length_threshold,
            }
        });

        // TODO after initializing this state, we need to check that it is actually valid

        let t0 = config.t0;
        let dt = config.dt;
        let t_max = config.t_max;
        let save_interval = config.save_interval;
        let time = FixedStepsize::from_partial_save_interval(t0, dt, t_max, save_interval)
            .or_else(|x| Err(SimulationError::from(x)))?;
        let storage = StorageBuilder::new().priority(config.storage_priority.clone());
        let settings = Settings {
            n_threads: config.n_threads,
            time,
            storage,
            show_progressbar: config.show_progressbar,
        };

        let mut domain = CartesianCuboid::from_boundaries_and_n_voxels(
            [0.0; 3],
            [config.domain_size, config.domain_size, 5.0],
            [config.n_voxels, config.n_voxels, 1],
        )
        .or_else(|x| Err(SimulationError::from(x)))?;
        domain.rng_seed = config.rng_seed;
        let domain = CartesianCuboidRods { domain };

        let storage_access = run_simulation!(
            agents: bacteria,
            domain: domain,
            settings: settings,
            aspects: [Mechanics, Interaction, Cycle],
        )?;
        let all_agents = storage_access
            .cells
            .load_all_elements()
            .or_else(|x| Err(SimulationError::from(x)))?
            .into_iter()
            .map(|(iteration, cells)| {
                (
                    iteration,
                    cells
                        .into_iter()
                        .map(|(identifier, (cell, _))| (identifier, (cell.cell, cell.parent)))
                        .collect::<std::collections::HashMap<_, _>>(),
                )
            })
            .collect::<std::collections::HashMap<_, _>>();
        Ok(all_agents)
    })
}

/// Sorts an iterator of [CellIdentifier] deterministically.
#[pyfunction]
pub fn sort_cellular_identifiers(
    identifiers: Vec<CellIdentifier>,
) -> Result<Vec<CellIdentifier>, PyErr> {
    let mut identifiers = identifiers;
    identifiers.sort();
    Ok(identifiers)
}
