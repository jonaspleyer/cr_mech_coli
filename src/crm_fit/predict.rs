use super::settings::*;
use crate::{PhysInt, PhysicalInteraction};
use cellular_raza::prelude::{MiePotentialF32, MorsePotentialF32, RodInteraction};
use numpy::{PyArrayMethods, ToPyArray};
use pyo3::prelude::*;

/// TODO
#[pyfunction]
pub fn run_simulation(
    py: Python,
    parameters: Vec<f32>,
    initial_positions: numpy::PyReadonlyArray3<f32>,
    settings: &Settings,
) -> PyResult<crate::CellContainer> {
    let config = settings.to_config(py)?;
    let mut positions = initial_positions.as_array().to_owned();

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
            .add_assign(settings.domain_height() / 2.0);
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
    } = settings.parameters.extract(py)?;

    let constants: Constants = settings.constants.extract(py)?;

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
                        if individual == Some(true) {
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
                            bound,
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
                growth_rate_distr: (growth_rate[n], 0.),
                spring_length_threshold: f32::INFINITY,
                neighbor_reduction: None,
            }
        })
        .collect();
    Ok(crate::run_simulation_with_agents(&config, agents)?)
}
