//! We fix the y-coordiante which corresponds to index 1.
//! ```
//! # let domain_size = 10.0;
//! # let pos = [0., 10.0/2.0, 0.];
//! assert!(pos[2] == domain_size / 2.0);
//! ```
//! Then we can use the gel_pressure which acts from top to bottom without any modifications.

use core::f32;
use itertools::Itertools;
use pyo3::prelude::*;

use crate::{
    GrowthRateSetter, PhysInt, PhysicalInteraction, RodAgent, SpringLengthThresholdSetter,
};
use approx::AbsDiffEq;
use cellular_raza::prelude::*;
use serde::{Deserialize, Serialize};

short_default::default! {
    /// Contains all parameters required for the simulation
    #[derive(Clone, Debug, PartialEq, Deserialize, Serialize, AbsDiffEq)]
    #[pyclass(get_all, set_all)]
    pub struct Parameters {
        /// Overall Domain Size
        domain_size: f32 = 200.0,
        /// Size for which to block movement along additional coordinate
        block_size: f32 = 30.0,
        /// Overall width of the tube
        tube_width: f32 = 100.0,
        /// Drag force exerted by the flow
        drag_force: f32 = 0.1,
        /// Maximum simulation time
        t_max: f32 = 150.0,
        /// Interval in which to save Agent data
        save_interval: f32 = 5.0,
        /// Time increment for solving the equations
        dt: f32 = 0.1,
        /// Overall starting length of the rod
        rod_length: f32 = 24.0,
        /// Rigidity of the rod
        rod_rigidity: f32 = 2.0,
        /// Tension with which the segment lengths are enforced
        spring_tension: f32 = 0.5,
        /// Growth rate of the rod
        growth_rate: f32 = 0.1,
        /// Damping constant
        damping: f32 = 0.05,
        /// Number of vertices to use for Rod
        #[approx(equal)]
        n_vertices: usize = 8,
        /// Show/hide progressbar during solving of simulation
        #[approx(equal)]
        progressbar: Option<String> = None,
    }
}

#[pymethods]
impl Parameters {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    fn __repr__(&self) -> String {
        format!("{:#?}", self)
    }
}

/// RodAgent which is based on the :class:`RodAgent` class.
///
/// It applies a fixed value to the first vertex of the rod and ensures that vertices can only move
/// along a one-dimensional line when inside the nanochamber given by the `block_size` value.
#[derive(Clone, Debug, CellAgent, Deserialize, Serialize)]
#[pyclass]
pub struct FixedRod {
    #[pyo3(get)]
    #[InteractionInformation]
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
    #[inline]
    fn pos(&self) -> RodPos {
        self.agent.mechanics.pos()
    }

    #[inline]
    fn set_pos(&mut self, pos: &RodPos) {
        let mut new_pos = pos.clone();
        // Make sure that lower positions are blocked
        new_pos.row_mut(0)[0] = 0.0;
        (1..new_pos.nrows()).for_each(|n| {
            let x: f32 = new_pos.row(n - 1)[0];
            if x <= self.block_size {
                new_pos.row_mut(n)[2] = self.domain_size / 2.0;
            }
        });
        self.agent.mechanics.set_pos(&new_pos);
    }
}

impl Velocity<RodPos> for FixedRod {
    #[inline]
    fn velocity(&self) -> RodPos {
        self.agent.velocity()
    }

    #[inline]
    fn set_velocity(&mut self, velocity: &RodPos) {
        let mut new_velocity = velocity.clone();
        new_velocity.row_mut(0)[2] = 0.0;
        new_velocity
            .row_iter_mut()
            .enumerate()
            .skip(1)
            .for_each(|(n, mut col)| {
                if self.pos()[(n - 1, 0)] <= self.block_size {
                    col[2] = 0.0;
                }
            });
        self.agent.mechanics.set_velocity(&new_velocity);
    }
}

impl Intracellular<f32> for FixedRod {
    fn get_intracellular(&self) -> f32 {
        self.agent.mechanics.spring_length
    }

    fn set_intracellular(&mut self, intracellular: f32) {
        self.agent.mechanics.spring_length = intracellular;
    }
}

impl Cycle<FixedRod, f32> for FixedRod {
    fn update_cycle(
        rng: &mut rand_chacha::ChaCha8Rng,
        dt: &f32,
        cell: &mut FixedRod,
    ) -> Option<CycleEvent> {
        RodAgent::update_cycle(rng, dt, &mut cell.agent)?;
        None
    }
    fn divide(_: &mut rand_chacha::ChaCha8Rng, _: &mut Self) -> Result<Self, DivisionError> {
        Err(DivisionError("This function should never be called".into()))
    }
}

struct MyDomain {
    domain: CartesianCuboidRods<f32, 3>,
    block_size: f32,
    tube_width: f32,
}

impl Domain<FixedRod, MySubDomain> for MyDomain {
    type VoxelIndex = <CartesianCuboidRods<f32, 3> as Domain<
        FixedRod,
        CartesianSubDomainRods<f32, 3>,
    >>::VoxelIndex;
    type SubDomainIndex = <CartesianCuboidRods<f32, 3> as Domain<
        FixedRod,
        CartesianSubDomainRods<f32, 3>,
    >>::SubDomainIndex;

    fn decompose(
        self,
        n_subdomains: core::num::NonZeroUsize,
        cells: Vec<FixedRod>,
    ) -> Result<DecomposedDomain<Self::SubDomainIndex, MySubDomain, FixedRod>, DecomposeError> {
        let DecomposedDomain {
            n_subdomains,
            index_subdomain_cells,
            neighbor_map,
            rng_seed,
        } = self.domain.decompose(n_subdomains, cells)?;
        let block_size = self.block_size;
        let tube_width = self.tube_width;
        let index_subdomain_cells = index_subdomain_cells
            .into_iter()
            .map(|(index, subdomain, cells)| {
                (
                    index,
                    MySubDomain {
                        subdomain,
                        block_size,
                        tube_width,
                    },
                    cells,
                )
            })
            .collect();
        Ok(DecomposedDomain {
            n_subdomains,
            index_subdomain_cells,
            neighbor_map,
            rng_seed,
        })
    }
}

#[derive(Clone, Serialize, Deserialize, SubDomain)]
struct MySubDomain {
    #[Base]
    #[Mechanics]
    #[SortCells]
    subdomain: CartesianSubDomainRods<f32, 3>,
    block_size: f32,
    tube_width: f32,
}

impl SubDomainForce<RodPos, RodPos, RodPos, f32> for MySubDomain {
    #[inline]
    fn calculate_custom_force(
        &self,
        pos: &RodPos,
        _: &RodPos,
        _: &f32,
    ) -> Result<RodPos, CalcError> {
        let mut force = RodPos::zeros(pos.nrows());
        pos.row_iter()
            .enumerate()
            .tuple_windows()
            .for_each(|((n1, p1), (n2, p2))| {
                let length = (p2 - p1).norm();
                let dir = (p2 - p1).normalize();
                let angle = dir.angle(&nalgebra::matrix![0.0, 1.0, 0.0]);
                let y = ((p1[0] + p2[0]) / 2.0 - self.block_size).max(0.0);
                let r = self.tube_width;
                let f = self.subdomain.gel_pressure * length * angle.sin() * (y * r - y.powi(2));
                force.row_mut(n1)[2] -= f / 2.0;
                force.row_mut(n2)[2] -= f / 2.0;
            });
        Ok(force)
    }
}

#[pyfunction]
#[pyo3(signature = (parameters, initial_pos=None))]
fn run_sim(
    parameters: Parameters,
    initial_pos: Option<Bound<numpy::PyArray2<f32>>>,
) -> PyResult<Vec<(u64, FixedRod)>> {
    let domain_size = parameters.domain_size;

    let mechanics = RodMechanics {
        pos: nalgebra::MatrixXx3::zeros(parameters.n_vertices),
        vel: nalgebra::MatrixXx3::zeros(parameters.n_vertices),
        diffusion_constant: 0.0,
        spring_tension: parameters.spring_tension,
        rigidity: parameters.rod_rigidity,
        spring_length: parameters.rod_length / (parameters.n_vertices as f32 - 1.0),
        damping: parameters.damping,
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
    let position = if let Some(initial_pos) = initial_pos {
        use numpy::PyArrayMethods;
        let nrows = parameters.n_vertices;
        let iter: Vec<f32> = initial_pos.to_vec()?;
        nalgebra::MatrixXx3::<f32>::from_iterator(nrows, iter)
    } else {
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
            growth_rate: parameters.growth_rate,
            growth_rate_setter: GrowthRateSetter::NormalDistr {
                mean: parameters.growth_rate,
                std: 0.,
            },
            spring_length_threshold: f32::INFINITY,
            spring_length_threshold_setter: SpringLengthThresholdSetter::Explicit {
                l1: f32::INFINITY,
                l2: f32::INFINITY,
            },
            neighbor_reduction: None,
        },
        domain_size,
        block_size: parameters.block_size,
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
        progressbar: parameters.progressbar,
    };

    let domain_size = [domain_size, 0.1, domain_size];
    let domain = CartesianCuboid::from_boundaries_and_n_voxels([0.0; 3], domain_size, [1, 1, 1])
        .map_err(SimulationError::from)?;
    let domain = MyDomain {
        domain: CartesianCuboidRods {
            domain,
            gel_pressure: parameters.drag_force,
            surface_friction: 0.0,
            surface_friction_distance: f32::INFINITY,
        },
        block_size: parameters.block_size,
        tube_width: parameters.tube_width,
    };

    let storage = cellular_raza::prelude::run_simulation!(
        agents: agents,
        domain: domain,
        settings: settings,
        aspects: [Mechanics, Cycle, DomainForce],
        zero_force_default: |c: &FixedRod| {
            nalgebra::MatrixXx3::zeros(c.agent.mechanics.pos().nrows())
        },
    )?;
    let cells: Vec<_> = storage
        .cells
        .load_all_elements()
        .map_err(SimulationError::from)?
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

#[pyfunction]
#[pyo3(signature = (parameters, t_relax, initial_pos=None))]
fn run_sim_with_relaxation(
    py: Python,
    parameters: &Parameters,
    t_relax: f32,
    initial_pos: Option<Bound<numpy::PyArray2<f32>>>,
) -> PyResult<Vec<(u64, FixedRod)>> {
    // Run first iteration of simulation
    let mut rods = run_sim(parameters.clone(), initial_pos)?;
    // Obtain final position of rod
    let p = rods[1].1.agent.mechanics.pos.clone();
    let initial_pos =
        ndarray::Array2::<f32>::from_shape_vec(p.shape(), p.into_iter().copied().collect())
            .unwrap();
    // Set final position of rod as initial position for second run
    let initial_pos = numpy::PyArray2::from_owned_array(py, initial_pos);
    let mut new_parameters = parameters.clone();
    // Remove the drag force and run simulation again
    new_parameters.drag_force = 0.0;
    new_parameters.t_max = t_relax;
    let (iter_final, agent) = run_sim(new_parameters, Some(initial_pos))?[1].clone();

    // Append last rods to results
    rods.push((rods[1].0 + iter_final, agent));
    assert_eq!(rods.len(), 3);
    Ok(rods)
}

/// A Python module implemented in Rust.
pub fn crm_amir(py: Python) -> PyResult<Bound<PyModule>> {
    let m = PyModule::new(py, "cr_mech_coli.crm_amir.crm_amir_rs")?;
    m.add_function(wrap_pyfunction!(run_sim, &m)?)?;
    m.add_function(wrap_pyfunction!(run_sim_with_relaxation, &m)?)?;
    m.add_class::<FixedRod>()?;
    m.add_class::<Parameters>()?;
    Ok(m)
}
