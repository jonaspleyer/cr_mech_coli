use cellular_raza::prelude::{MorsePotentialF32, RodInteraction, RodMechanics, StorageOption};
use cr_mech_coli::{
    _generate_positions, run_simulation_with_agents, Configuration, PhysInt, PhysicalInteraction,
    RodAgent,
};

fn main() {
    let n_vertices = 8;
    let n_agents = 16;
    let domain_size = (n_agents as f32).sqrt() * 75.;

    let config = Configuration {
        storage_options: vec![StorageOption::Memory],
        progressbar: Some(String::new()),
        n_threads: 4.try_into().unwrap(),
        domain_size: [domain_size; 2],
        n_voxels: [4, 4],
        t_max: 150.0,
        ..Default::default()
    };

    let mechanics = RodMechanics {
        pos: nalgebra::MatrixXx3::zeros(n_vertices),
        vel: nalgebra::MatrixXx3::zeros(n_vertices),
        diffusion_constant: 0.0,
        spring_tension: 1.0,
        rigidity: 2.0,
        spring_length: 3.0,
        damping: 1.0,
    };
    let mechanics_settings = cr_mech_coli::RodMechanicsSettings {
        pos: mechanics.pos.clone(),
        vel: mechanics.vel.clone(),
        diffusion_constant: mechanics.diffusion_constant,
        spring_tension: mechanics.spring_tension,
        rigidity: mechanics.rigidity,
        spring_length: mechanics.spring_length,
        damping: mechanics.damping,
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
    let positions = _generate_positions(
        n_agents,
        &mechanics_settings,
        &config,
        0,
        [0.0; 2],
        0.1,
        n_vertices,
    );

    let agents = positions
        .into_iter()
        .map(|p| {
            let mut agent = RodAgent {
                mechanics: mechanics.clone(),
                interaction: interaction.clone(),
                growth_rate: 0.1,
                growth_rate_setter: cr_mech_coli::GrowthRateSetter::NormalDistr {
                    mean: 0.1,
                    std: 0.,
                },
                spring_length_threshold: 6.0,
                spring_length_threshold_setter:
                    cr_mech_coli::SpringLengthThresholdSetter::Explicit { l1: 6.0, l2: 6.0 },
                neighbor_reduction: None,
            };
            <RodAgent as cellular_raza::concepts::Position<_>>::set_pos(&mut agent, &p);
            agent
        })
        .collect();
    run_simulation_with_agents(&config, agents).unwrap();
}
