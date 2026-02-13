"""
Synthetic Microscope Image Generation.

A submodule for cr_mech_coli that provides:

- Synthetic microscope image generation pipeline
- Image cloning from real microscope images
- Parameter optimization to match real images

.. code-block:: text
    :caption: Usage of the crm_gen script

    usage: crm_gen [-h] [--config CONFIG] {run,clone,fit} ...

    Synthetic Microscope Image Generation for cr_mech_coli

    positional arguments:
    {run,clone,fit}         Subcommand
        run                 Run the synthetic image generation pipeline
        clone               Clone a real microscope image to synthetic
        fit                 Optimize parameters to match real microscope images

    options:
      -h, --help            show this help message and exit
      --config CONFIG, -c CONFIG
                            Path to TOML configuration file


    ------------------------------------------------------------------------



    usage: crm_gen run [-h]

    Run bacteria growth simulation and generate synthetic microscope images. All parameters come
    from the TOML config file.

    options:
      -h, --help  show this help message and exit


    ------------------------------------------------------------------------



    usage: crm_gen clone [-h] [--output OUTPUT] [--n-vertices N_VERTICES] [--seed SEED]
                      microscope_image segmentation_mask

    Create a synthetic version of a real microscope image using cell positions extracted from a
    segmentation mask. Imaging parameters come from the TOML config file.

    positional arguments:
      microscope_image      Path to real microscope image (TIF)
      segmentation_mask     Path to segmentation mask (TIF)

    options:
      -h, --help            show this help message and exit
      --output OUTPUT, -o OUTPUT
                            Output directory (default: ./synthetic_output)
      --n-vertices N_VERTICES
                            Number of vertices per cell (overrides config, default: 8)
      --seed SEED           Random seed (overrides config)


    ------------------------------------------------------------------------



    usage: crm_gen fit [-h]

    Optimize synthetic image generation parameters to match real microscope images using
    differential evolution. All parameters come from the TOML config file ([optimization] section).

    options:
      -h, --help  show this help message and exit
"""

# Core scene generation
from .scene import (
    create_synthetic_scene,
    apply_synthetic_effects,
)

# Pipeline
from .pipeline import (
    run_pipeline,
    run_simulation,
    compute_cell_ages,
)

# Config
from .config import (
    load_config,
    get_default_config,
    get_default_config_path,
    PARAM_NAMES,
    DEFAULT_OPTIMIZATION_BOUNDS,
    DEFAULT_METRIC_WEIGHTS,
    DEFAULT_REGION_WEIGHTS,
)

# Background generation
from .background import (
    generate_phase_contrast_background,
)

# Filters and effects
from .filters import (
    apply_psf_blur,
    apply_halo_effect,
    apply_microscope_effects,
    apply_phase_contrast_pipeline,
    add_poisson_noise,
    add_gaussian_noise,
)

# Bacteria brightness
from .bacteria import (
    apply_original_brightness,
    apply_age_based_brightness,
    extract_original_brightness,
)

# Metrics
from .metrics import (
    compute_all_metrics,
    compute_ssim,
    compute_psnr,
    compute_color_distribution,
    load_image,
    plot_metrics,
)

# CLI entry point
from .main import crm_gen_main

__version__ = "0.1.0"
__all__ = [
    # Scene
    'create_synthetic_scene',
    'apply_synthetic_effects',
    # Pipeline
    'run_pipeline',
    'run_simulation',
    'compute_cell_ages',
    'load_config',
    # Background
    'generate_phase_contrast_background',
    # Filters
    'apply_psf_blur',
    'apply_halo_effect',
    'apply_microscope_effects',
    'apply_phase_contrast_pipeline',
    'add_poisson_noise',
    'add_gaussian_noise',
    # Bacteria
    'apply_original_brightness',
    'apply_age_based_brightness',
    'extract_original_brightness',
    # Metrics
    'compute_all_metrics',
    'compute_ssim',
    'compute_psnr',
    'compute_color_distribution',
    'load_image',
    'plot_metrics',
    # CLI
    'crm_gen_main',
]
