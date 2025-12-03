use super::settings::*;
use crate::{
    run_simulation_with_agents, PhysInt, PhysicalInteraction, SpringLengthThresholdSetter,
};
use cellular_raza::prelude::{
    CellIdentifier, MiePotentialF32, MorsePotentialF32, RodInteraction, SimulationError,
};
use pyo3::prelude::*;

#[pyfunction]
pub fn run_simulation(
    py: Python,
    parameters: Vec<f32>,
    initial_positions: numpy::PyReadonlyArray3<f32>,
    settings: &Settings,
) -> PyResult<crate::CellContainer> {
    let initial_positions = initial_positions.as_array();
    let config = settings.to_config(py)?;
    let parameter_defs = settings.parameters.extract(py)?;
    let constants: Constants = settings.constants.extract(py)?;
    let agents = define_initial_agents(
        parameters,
        initial_positions,
        settings.domain_height(),
        &parameter_defs,
        &constants,
    );
    Ok(run_simulation_with_agents(&config, agents)?)
}

/// TODO
pub fn define_initial_agents(
    parameters: Vec<f32>,
    initial_positions: numpy::ndarray::ArrayView3<f32>,
    domain_height: f32,
    parameter_defs: &Parameters,
    constants: &Constants,
) -> Vec<crate::RodAgent> {
    let mut positions = initial_positions.to_owned();

    // If the positions do not have dimension (?,?,3), we bring them to this dimension
    if positions.shape()[2] != 3 {
        let mut new_positions =
            numpy::ndarray::Array3::<f32>::zeros((positions.shape()[0], positions.shape()[1], 3));
        new_positions
            .slice_mut(numpy::ndarray::s![.., .., ..2])
            .assign(&positions.slice(numpy::ndarray::s![.., .., ..2]));
        use core::ops::AddAssign;
        new_positions
            .slice_mut(numpy::ndarray::s![.., .., 2])
            .add_assign(domain_height / 2.0);
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
    } = parameter_defs;

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
                            cutoff: constants.cutoff,
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
                            cutoff: constants.cutoff,
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
        res / (constants.n_vertices.get() - 1) as f32
    };

    let agents = positions
        .axis_iter(numpy::ndarray::Axis(0))
        .enumerate()
        .map(|(n, pos)| {
            let pos = nalgebra::Matrix3xX::<f32>::from_iterator(
                constants.n_vertices.get(),
                pos.iter().copied(),
            );
            let spring_length = pos_to_spring_length(&pos.transpose());
            crate::RodAgent {
                mechanics: cellular_raza::prelude::RodMechanics {
                    pos: pos.transpose(),
                    vel: nalgebra::MatrixXx3::zeros(constants.n_vertices.get()),
                    diffusion_constant: 0.0,
                    spring_tension: spring_tension[n],
                    rigidity: rigidity[n],
                    spring_length,
                    damping: damping[n],
                },
                interaction: interaction[n].clone(),
                growth_rate: growth_rate[n],
                growth_rate_setter: crate::GrowthRateSetter::NormalDistr {
                    mean: growth_rate[n],
                    std: 0.0,
                },
                spring_length_threshold: f32::INFINITY,
                spring_length_threshold_setter: SpringLengthThresholdSetter::Explicit {
                    l1: f32::INFINITY,
                    l2: f32::INFINITY,
                },
                neighbor_reduction: None,
            }
        })
        .collect();
    agents
}

#[pyfunction]
pub fn predict_calculate_cost(
    py: Python,
    parameters: Vec<f32>,
    positions_all: numpy::PyReadonlyArray4<f32>,
    iterations: Vec<usize>,
    settings: Settings,
) -> PyResult<f32> {
    let positions_all = positions_all.as_array();
    // let initial_positions = initial_positions.as_array();
    let config = settings.to_config(py)?;
    let parameter_defs = &settings.parameters.borrow(py);
    let constants = &settings.constants.borrow(py);
    let res = predict_calculate_cost_rs(
        parameters,
        positions_all,
        settings.domain_height(),
        parameter_defs,
        constants,
        &config,
        &iterations,
    )?;
    if !res.is_normal() {
        Ok(constants.error_cost)
    } else {
        Ok(res)
    }
}

pub fn predict_calculate_cost_rs(
    parameters: Vec<f32>,
    positions_all: numpy::ndarray::ArrayView4<f32>,
    domain_height: f32,
    parameter_defs: &Parameters,
    constants: &Constants,
    config: &crate::Configuration,
    iterations_images: &[usize],
) -> Result<f32, SimulationError> {
    let initial_positions = positions_all.slice(numpy::ndarray::s![0, .., .., ..]);
    let agents = define_initial_agents(
        parameters,
        initial_positions,
        domain_height,
        parameter_defs,
        constants,
    );
    let res = crate::run_simulation_with_agents(config, agents);
    let container = match res {
        Err(e) => {
            log::warn!("Encountered error during solving of system");
            log::warn!("{e}");
            return Ok(f32::NAN);
        }
        Ok(c) => c,
    };

    let mut total_cost = 0f32;
    let all_iterations = container.get_all_iterations();
    for (n_result, data_index) in iterations_images.iter().enumerate() {
        let it = all_iterations[*data_index];

        container
            .get_cells_at_iteration(it)
            .into_iter()
            .try_for_each(|(ident, (c, _))| {
                if let CellIdentifier::Initial(n_agent) = ident {
                    let p1 =
                        ndarray::Array2::from_shape_fn((c.mechanics.pos.nrows(), 2), |(i, j)| {
                            c.mechanics.pos[(i, j)]
                        });
                    let p2 = positions_all.slice(numpy::ndarray::s![n_result, n_agent, .., ..]);
                    total_cost += (p1.to_owned() - p2).map(|x| x.powf(2.0)).sum();
                    Ok(())
                } else {
                    Err(SimulationError::IndexError(
                        cellular_raza::prelude::IndexError(
                            "Cell division is currently not supported by this optimization\
                            algorithm"
                                .to_string(),
                        ),
                    ))
                }
            })?;
    }
    Ok(total_cost)
}
