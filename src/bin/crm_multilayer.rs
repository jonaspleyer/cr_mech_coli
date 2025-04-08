use cellular_raza::prelude::{MiePotentialF32, RodInteraction, RodMechanics};
use cr_mech_coli::{CellContainer, Configuration, PhysicalInteraction, RodAgent};
use pyo3::{prelude::*, types::PyDict, BoundObject};
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;

use clap::Parser;
use serde::{Deserialize, Serialize};

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    /// Path to the config file
    #[arg(
        short,
        long,
        value_name = "CONFIG_FILE",
        default_value_t = format!("")
    )]
    config_file: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(from = "MultilayerConfigSerde", into = "MultilayerConfigSerde")]
struct MultilayerConfig {
    base: Configuration,
    n_vertices: usize,
    // agent_settings: AgentSettings,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct Config {
    #[serde(flatten)]
    base: Configuration,
    n_vertices: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct MultilayerConfigSerde {
    config: Config,
    // agent_settings: AgentSettings,
}

impl From<MultilayerConfig> for MultilayerConfigSerde {
    fn from(value: MultilayerConfig) -> Self {
        let MultilayerConfig {
            base,
            n_vertices,
            // agent_settings,
        } = value;
        Self {
            config: Config { base, n_vertices },
            // agent_settings,
        }
    }
}

impl From<MultilayerConfigSerde> for MultilayerConfig {
    fn from(value: MultilayerConfigSerde) -> Self {
        let MultilayerConfigSerde {
            config,
            // agent_settings,
        } = value;
        Self {
            base: config.base,
            n_vertices: config.n_vertices,
            // agent_settings,
        }
    }
}

impl Default for MultilayerConfig {
    fn default() -> Self {
        let base = Configuration {
            gravity: 0.01,
            ..Default::default()
        };
        Self {
            base,
            n_vertices: 8,
            /* agent_settings: AgentSettings {
                mechanics: todo!(),
                interaction: todo!(),
                growth_rate: todo!(),
                spring_length_threshold: todo!(),
            },*/
        }
    }
}

fn plot_all_iteration(container: &CellContainer, config: &MultilayerConfig) -> PyResult<()> {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let cr_mech_coli = PyModule::import(py, "cr_mech_coli")?;
        let render_settings = cr_mech_coli.getattr("RenderSettings")?.call0()?;
        let kwargs = PyDict::from_sequence(
            &[
                (
                    "cell_container".into_pyobject(py)?,
                    container.clone().into_pyobject(py)?.as_any(),
                ),
                (
                    "domain_size".into_pyobject(py)?,
                    config.base.domain_size.into_pyobject(py)?.as_any(),
                ),
                (
                    "render_settings".into_pyobject(py)?,
                    render_settings.as_any(),
                ),
                (
                    "save_dir".into_pyobject(py)?,
                    "out/crm_multilayer".into_pyobject(py)?.as_any(),
                ),
                (
                    "render_raw_pv".into_pyobject(py)?,
                    true.into_pyobject(py)?.as_any(),
                ),
                (
                    "show_progressbar".into_pyobject(py)?,
                    true.into_pyobject(py)?.as_any(),
                ),
            ]
            .into_pyobject(py)?,
        )?
        .into_bound();
        cr_mech_coli.call_method("store_all_images", (), Some(&kwargs))?;
        Result::<_, PyErr>::Ok(())
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let config: MultilayerConfig = {
        let content = std::fs::read_to_string(&cli.config_file);
        match content {
            Ok(content) => toml::from_str::<MultilayerConfigSerde>(&content)?.into(),
            Err(e) => {
                println!(
                    "Encountered error {e} while reading file {}. Proceed with default values.",
                    cli.config_file
                );
                MultilayerConfig::default()
            }
        }
    };

    let n_vertices = config.n_vertices;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(config.base.rng_seed);
    let agents = vec![RodAgent {
        mechanics: RodMechanics {
            pos: nalgebra::MatrixXx3::from_fn(n_vertices, |n, m| {
                let dx = 1.0;
                if m == 0 {
                    config.base.domain_size[0] / 2.0 + n as f32 * dx - n_vertices as f32 / 2. * dx
                } else if m == 1 {
                    config.base.domain_size[1] / 2.0 + rng.random_range(-0.01..0.01)
                } else {
                    config.base.domain_height / 2.0
                }
            }),
            vel: nalgebra::MatrixXx3::zeros(n_vertices),
            spring_tension: 3.0,
            rigidity: 6.0,
            spring_length: 1.0,
            damping: 0.5,
            diffusion_constant: 0.1,
        },
        interaction: RodInteraction(PhysicalInteraction::MiePotentialF32(MiePotentialF32 {
            radius: 1.0,
            strength: 0.02,
            bound: 0.5,
            cutoff: 3.0,
            en: 6.41,
            em: 6.19,
        })),
        growth_rate: 0.06,
        spring_length_threshold: 1.5,
    }];

    let container = cr_mech_coli::run_simulation_with_agents(&config.base, agents)?;

    for (iteration, cells) in container.get_cells().into_iter() {
        println!("{iteration:8}: {:8}", cells.len());
    }
    plot_all_iteration(&container, &config).unwrap();
    Ok(())
}
