use cellular_raza::prelude::*;
use serde::{Deserialize, Serialize};

pub const N_ROD_SEGMENTS: usize = 8;

#[derive(CellAgent, Clone, Debug, Deserialize, Serialize)]
pub struct RodAgent {
    #[Mechanics]
    mechanics: RodMechanics<f32, N_ROD_SEGMENTS, 2>,
    #[Interaction]
    interaction: RodInteraction<MorsePotentialF32>,
}

fn main() -> Result<(), SimulationError> {
    tracing_subscriber::fmt::init();

    use rand::Rng;
    use rand_chacha::rand_core::SeedableRng;
    tracing::info!("Here Rusty 1");
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
    let spring_length = 5.0; // MICROMETRE
    let dx = spring_length * N_ROD_SEGMENTS as f32;
    let bacteria = (0..2).map(|_| {
        let p1 = [
            rng.gen_range(dx..10.0 * dx - dx),
            rng.gen_range(dx..10.0 * dx - dx),
        ];
        let angle: f32 = rng.gen_range(0.0..2.0 * std::f32::consts::PI);
        RodAgent {
            mechanics: RodMechanics {
                pos: nalgebra::SMatrix::<f32, N_ROD_SEGMENTS, 2>::from_fn(|r, c| {
                    p1[0]
                        + r as f32 * spring_length * if c == 0 { angle.cos() } else { angle.sin() }
                }),
                vel: nalgebra::SMatrix::<f32, N_ROD_SEGMENTS, 2>::from_fn(|_, _| 0.0),
                diffusion_constant: 0.0,
                spring_tension: 0.1,
                angle_stiffness: 0.05,
                spring_length: 1.0,
                damping: 0.1,
            },
            interaction: RodInteraction(MorsePotentialF32 {
                cutoff: 3.0,
                potential_width: 1.5,
                radius: 5.0,
                strength: 0.5,
            }),
        }
    });
    tracing::info!("Here Rusty 2");

    let t0 = 0.0;
    let dt = 0.05;
    let t_max = 10.0;
    let save_interval = 0.25;
    let time = FixedStepsize::from_partial_save_interval(t0, dt, t_max, save_interval)?;
    let storage = StorageBuilder::new().priority([StorageOption::Memory]);
    let settings = Settings {
        n_threads: 1.try_into().unwrap(),
        time,
        storage,
        show_progressbar: true,
    };
    tracing::info!("Here Rusty 3");

    let mut domain = CartesianCuboid::from_boundaries_and_n_voxels([0.0; 2], [100.0; 2], [1; 2])
        .or_else(|x| Err(SimulationError::from(x)))?;
    domain.rng_seed = 1;
    let domain = CartesianCuboidRods { domain };
    tracing::info!("Here Rusty 4");

    run_simulation!(
        agents: bacteria,
        domain: domain,
        settings: settings,
        aspects: [Mechanics, Interaction],
    )?;
    tracing::info!("Here Rusty 6");
    Ok(())
}
