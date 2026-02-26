# crm_gen

Synthetic Microscope Image Generation submodule for cr_mech_coli.

## Overview

This module provides tools for generating realistic synthetic phase contrast
microscope images of bacteria, using the cr_mech_coli simulation framework.
It includes:

- **Bacteria growth simulation** with configurable parameters
- **Synthetic image generation** with realistic microscope effects
- **Image cloning** — recreate real microscope images as synthetic versions
- **Parameter optimisation** to match real microscope images

## CLI Usage

After installation, one command with three subcommands is available.
Each subcommand has an optional `--config` argument; positional arguments
come first, `--config` always last.

### crm_gen run

Run the full synthetic image generation pipeline:

```bash
crm_gen run                          # use default generation config
crm_gen run --config my_gen.toml     # custom generation config
```

Runs a bacteria growth simulation and generates synthetic microscope images.
Uses a *generation config* (`configs/default_gen_config.toml` by default).

### crm_gen clone

Clone a real microscope image to synthetic:

```bash
crm_gen clone image.tif mask.tif
crm_gen clone image.tif mask.tif --output ./output --config my_gen.toml
```

Extracts cell positions from the segmentation mask and recreates the image
synthetically using imaging parameters from the generation config.

### crm_gen fit

Optimise synthetic image parameters to match real microscope images:

```bash
crm_gen fit path/to/real/images/
crm_gen fit path/to/real/images/ --config my_fit.toml
```

Uses differential evolution to find optimal imaging parameters for matching
real microscope images. The input directory is a required positional argument.
Uses a *fit config* (`configs/default_fit_config.toml` by default) — this
contains only optimisation hyperparameters; the imaging parameters themselves
are the *output* of the fit.

## Configuration

`run` and `clone` use a **generation config** with the following sections:

- `[pipeline]` — Output settings, number of simulations and frames
- `[simulation]` — Physics parameters (growth rate, cell interactions)
- `[rendering]` — PyVista render settings
- `[synthetic]` — The 7 optimised imaging parameters (output of `fit`)
- `[background]` — Background generation settings
- `[halo]` — Phase contrast halo effect settings
- `[brightness]` — Cell brightness mode (age-based or original)

`fit` uses a **fit config** with the following sections:

- `[optimization]` — Differential evolution hyperparameters and output settings
- `[optimization.bounds]` — Search bounds `[min, max]` for each imaging parameter
- `[optimization.metric_weights]` — Weights for SSIM, PSNR, histogram distance
- `[optimization.region_weights]` — Foreground vs. background loss weighting

Default configs are in `configs/default_gen_config.toml` and
`configs/default_fit_config.toml`.

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

## Module Structure

```
crm_gen/
├── __init__.py               # Public API
├── main.py                   # CLI entry point (subparsers)
├── config.py                 # Configuration utilities
├── configs/
│   ├── default_gen_config.toml   # Default generation config (run, clone)
│   └── default_fit_config.toml   # Default fit config (fit)
├── pipeline.py               # Full simulation-to-image pipeline
├── scene.py                  # Single-frame compositing (uses background, bacteria, filters)
├── background.py             # Phase contrast background generation
├── filters.py                # Optical effects and sensor noise (PSF, Poisson, Gaussian)
├── bacteria.py               # Per-cell brightness assignment
├── metrics.py                # Image comparison metrics (SSIM, PSNR, histogram)
├── optimization.py           # Differential evolution parameter optimisation
└── visualization.py          # Diagnostic plots for optimisation results
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
