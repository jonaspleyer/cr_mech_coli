# crm_gen

Synthetic Microscope Image Generation submodule for cr_mech_coli.

## Overview

This module provides tools for generating realistic synthetic phase contrast microscope images of bacteria, using the cr_mech_coli simulation framework. It includes:

- **Bacteria growth simulation** with configurable parameters
- **Synthetic image generation** with realistic microscope effects
- **Image cloning** - recreate real microscope images as synthetic versions
- **Parameter optimization** to match real microscope images

## CLI Usage

After installation, one command with three subcommands is available:

### crm_gen run

Run the full synthetic image generation pipeline:

```bash
crm_gen --config my_config.toml run
```

This runs a bacteria growth simulation and generates synthetic microscope images with age-based brightness variations.

### crm_gen clone

Clone a real microscope image to synthetic:

```bash
crm_gen --config my_config.toml clone image.tif mask.tif --output ./output
```

This extracts cell positions from the mask and recreates the image synthetically using parameters from the config file.

### crm_gen fit

Optimize synthetic image parameters to match real images:

```bash
crm_gen --config my_config.toml fit
```

Uses differential evolution to find optimal parameters for matching real microscope images. Requires `[optimization] input_dir` in the config file.

## Python API

```python
from cr_mech_coli.crm_gen import (
    run_pipeline,
    create_synthetic_scene,
    apply_synthetic_effects,
    generate_phase_contrast_background,
)

# Run simulation pipeline
run_pipeline(
    output_dir="./outputs",
    n_frames=20,
    image_size=(1024, 1024),
    n_bacteria_range=(5, 10),
)

# Clone a single image
synthetic_img, synthetic_mask = create_synthetic_scene(
    microscope_image_path="real.tif",
    segmentation_mask_path="mask.tif",
    output_dir="./output",
)
```

## Configuration

All subcommands share a single TOML configuration file. See `default_config.toml` for all available options:

- `[pipeline]` - Output settings, number of simulations/frames
- `[simulation]` - Physics parameters (growth rate, interactions)
- `[rendering]` - PyVista render settings
- `[synthetic]` - The 7 optimized imaging parameters
- `[background]` - Background generation settings
- `[halo]` - Phase contrast halo effect settings
- `[brightness]` - Cell brightness mode (age-based or original)
- `[optimization]` - Differential evolution settings for `crm_gen fit`

## Module Structure

```
crm_gen/
├── __init__.py          # Public API
├── main.py              # CLI entry point (subparsers)
├── config.py            # Configuration utilities
├── default_config.toml  # Default configuration
├── pipeline.py          # Main pipeline
├── scene.py             # Scene generation
├── background.py        # Background generation
├── filters.py           # Microscope effects
├── bacteria.py          # Brightness transfer
├── metrics.py           # Image comparison metrics
├── optimization.py      # Parameter optimization
└── visualization.py     # Plotting utilities
```

## Dependencies

- numpy
- scipy
- tifffile
- matplotlib
- scikit-image
- tqdm
- tomli (Python < 3.11)
- cr_mech_coli (parent package)
