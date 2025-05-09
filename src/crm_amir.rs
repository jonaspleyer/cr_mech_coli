//! We fix the y-coordiante which corresponds to index 1.
//! ```
//! # let domain_size = 10.0;
//! # let pos = [0., 10.0/2.0, 0.];
//! assert!(pos[2] == domain_size / 2.0);
//! ```
//! Then we can use the gel_pressure which acts from top to bottom without any modifications.

use core::f32;
use pyo3::prelude::*;

use crate::{PhysInt, PhysicalInteraction, RodAgent};
use approx::AbsDiffEq;
use cellular_raza::prelude::*;
use serde::{Deserialize, Serialize};

short_default::default! {
    #[derive(Clone, Debug, PartialEq, Deserialize, Serialize, AbsDiffEq)]
    #[pyclass(get_all, set_all)]
    pub struct Parameters {
        /// Overall Domain Size
        domain_size: f32 = 200.0,
        /// Size for which to block movement along additional coordinate
        block_size: f32 = 30.0,
        /// Maximum simulation time
        t_max: f32 = 150.0,
        /// Interval in which to save Agent data
        save_interval: f32 = 5.0,
        /// Time increment for solving the equations
        dt: f32 = 0.1,
        /// Overall starting length of the rod
        rod_length: f32 = 24.0,
        /// Rigidity of the rod
        rod_rigiditiy: f32 = 2.0,
        /// Number of vertices to use for Rod
        #[approx(equal)]
        n_vertices: usize = 8,
    }
}

#[derive(Clone, Debug, CellAgent, Deserialize, Serialize)]
#[pyclass]
pub struct FixedRod {
    #[Interaction]
    #[pyo3(get)]
    agent: RodAgent,
    #[pyo3(get)]
    domain_size: f32,
    #[pyo3(get)]
    block_size: f32,
}

type RodPos = nalgebra::MatrixXx3<f32>;

impl Mechanics<RodPos, RodPos, RodPos, f32> for FixedRod {
    fn calculate_increment(&self, force: RodPos) -> Result<(RodPos, RodPos), CalcError> {
        self.agent.mechanics.calculate_increment(force)
    }

    fn get_random_contribution(
        &self,
        rng: &mut rand_chacha::ChaCha8Rng,
        dt: f32,
    ) -> Result<(RodPos, RodPos), RngError> {
        self.agent.mechanics.get_random_contribution(rng, dt)
    }
}

impl Position<RodPos> for FixedRod {
    fn pos(&self) -> RodPos {
        self.agent.mechanics.pos()
    }

    fn set_pos(&mut self, pos: &RodPos) {
        let mut new_pos = pos.clone();
        new_pos.row_iter_mut().for_each(|mut col| {
            if col[0] <= self.block_size {
                col[2] = self.domain_size / 2.0;
            }
        });
        self.agent.mechanics.set_pos(&new_pos);
    }
}

impl Velocity<RodPos> for FixedRod {
    fn velocity(&self) -> RodPos {
        self.agent.velocity()
    }

    fn set_velocity(&mut self, velocity: &RodPos) {
        self.agent.mechanics.set_velocity(velocity);
    }
}

impl Cycle<FixedRod, f32> for FixedRod {
    fn update_cycle(
        _: &mut rand_chacha::ChaCha8Rng,
        dt: &f32,
        cell: &mut Self,
    ) -> Option<CycleEvent> {
        cell.agent.mechanics.spring_length += cell.agent.growth_rate * dt;
        None
    }
    fn divide(_: &mut rand_chacha::ChaCha8Rng, _: &mut Self) -> Result<Self, DivisionError> {
        panic!()
    }
}

#[pyfunction]
fn run_sim(
    parameters: Parameters,
) -> Result<Vec<(u64, FixedRod)>, cellular_raza::prelude::SimulationError> {
    let domain_size = 200.0;

    let mechanics = RodMechanics {
        pos: nalgebra::MatrixXx3::zeros(parameters.n_vertices),
        vel: nalgebra::MatrixXx3::zeros(parameters.n_vertices),
        diffusion_constant: 0.0,
        spring_tension: 1.0,
        rigidity: parameters.rod_rigiditiy,
        spring_length: 3.0,
        damping: 1.0,
    };

    let interaction = RodInteraction(PhysicalInteraction(
        PhysInt::MorsePotentialF32(MorsePotentialF32 {
            radius: 3.0,
            potential_stiffness: 0.5,
            cutoff: 10.0,
            strength: 0.1,
        }),
        0,
    ));
    let position = {
        let mut pos = nalgebra::MatrixXx3::zeros(parameters.n_vertices);
        for (i, mut p) in pos.row_iter_mut().enumerate() {
            p[0] = i as f32 * mechanics.spring_length;
            p[1] = 2.0 * f32::EPSILON;
            p[2] = domain_size / 2.0;
        }
        pos
    };

    let agents = vec![FixedRod {
        agent: RodAgent {
            mechanics: RodMechanics {
                pos: position,
                ..mechanics
            },
            interaction,
            growth_rate: 0.1,
            growth_rate_distr: (0.1, 0.),
            spring_length_threshold: f32::INFINITY,
            neighbor_reduction: None,
        },
        domain_size,
        block_size: 30.0,
    }];

    let time = FixedStepsize::from_partial_save_interval(
        0.,
        parameters.dt,
        parameters.t_max,
        parameters.save_interval,
    )
    .map_err(SimulationError::from)?;
    let storage = StorageBuilder::new().priority([StorageOption::Memory]);
    let settings = Settings {
        n_threads: 1.try_into().unwrap(),
        time,
        storage,
        show_progressbar: true,
    };

    let domain_size = [domain_size, 0.1, domain_size];
    let domain = CartesianCuboid::from_boundaries_and_n_voxels([0.0; 3], domain_size, [1, 1, 1])?;
    let domain = CartesianCuboidRods {
        domain,
        gel_pressure: 0.1,
        surface_friction: 0.0,
        surface_friction_distance: f32::INFINITY,
    };

    let storage = cellular_raza::prelude::run_simulation!(
        agents: agents,
        domain: domain,
        settings: settings,
        aspects: [Mechanics, Interaction, Cycle, DomainForce],
        zero_force_default: |c: &FixedRod| {
            nalgebra::MatrixXx3::zeros(c.agent.mechanics.pos().nrows())
        },
    )?;
    let cells: Vec<_> = storage
        .cells
        .load_all_elements()?
        .into_iter()
        .map(|(iteration, cells)| {
            (
                iteration,
                cells
                    .into_iter()
                    .map(|(_, (cbox, _))| cbox.cell)
                    .next()
                    .unwrap(),
            )
        })
        .collect();
    Ok(cells)
}

/// A Python module implemented in Rust.
pub fn crm_amir(py: Python) -> PyResult<Bound<PyModule>> {
    let m = PyModule::new(py, "crm_amir")?;
    m.add_function(wrap_pyfunction!(run_sim, &m)?)?;
    m.add_class::<Parameters>()?;
    Ok(m)
}
