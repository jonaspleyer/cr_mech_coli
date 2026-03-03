from pathlib import Path
import pytest
import cr_mech_coli.crm_gen.config as _config_module
from cr_mech_coli.crm_gen.config import load_config

# Derive config paths from the installed package location so the tests work
# regardless of the working directory.
_CONFIGS_DIR = Path(_config_module.__file__).parent / "configs"
_GEN_CONFIG = _CONFIGS_DIR / "default_gen_config.toml"
_FIT_CONFIG = _CONFIGS_DIR / "default_fit_config.toml"


# Generation config

def test_load_default_gen_config():
    """Default generation config loads without errors."""
    config = load_config(str(_GEN_CONFIG))
    assert isinstance(config, dict)


def test_gen_config_top_level_sections():
    """All required top-level sections are present."""
    config = load_config(str(_GEN_CONFIG))
    required = {"pipeline", "simulation", "rendering", "synthetic", "background", "halo", "brightness"}
    missing = required - config.keys()
    assert not missing, f"Missing sections: {missing}"


def test_gen_config_synthetic_param_types():
    """
    Imaging parameters are numeric types.

    A TOML edit that accidentally turns  psf_sigma = 1.0  into
    psf_sigma = "1.0"  would only fail at runtime deep inside the pipeline;
    this test catches it immediately.
    """
    synth = load_config(str(_GEN_CONFIG))["synthetic"]

    float_params = [
        "bg_base_brightness",
        "bg_gradient_strength",
        "bac_halo_intensity",
        "psf_sigma",
        "peak_signal",
        "gaussian_sigma",
    ]
    int_params = ["bg_noise_scale"]

    for param in float_params:
        assert isinstance(synth[param], float), (
            f"synthetic.{param} should be float, got {type(synth[param])}"
        )
    for param in int_params:
        assert isinstance(synth[param], int), (
            f"synthetic.{param} should be int, got {type(synth[param])}"
        )


def test_gen_config_pipeline_section():
    """Pipeline section contains the expected keys."""
    pipeline = load_config(str(_GEN_CONFIG))["pipeline"]
    for key in ("output_dir", "n_simulations", "n_frames", "image_size", "seed"):
        assert key in pipeline, f"pipeline.{key} missing from gen config"


# Fit config

def test_load_default_fit_config():
    """Default fit config loads without errors."""
    config = load_config(str(_FIT_CONFIG))
    assert isinstance(config, dict)


def test_fit_config_optimization_section():
    """Optimization section and its required sub-keys are present."""
    opt = load_config(str(_FIT_CONFIG))["optimization"]
    for key in ("maxiter", "popsize", "workers", "seed", "bounds"):
        assert key in opt, f"optimization.{key} missing from fit config"


def test_fit_config_bounds_cover_all_params():
    """Bounds are defined for all seven imaging parameters."""
    from cr_mech_coli.crm_gen.config import PARAM_NAMES
    bounds = load_config(str(_FIT_CONFIG))["optimization"]["bounds"]
    missing = [p for p in PARAM_NAMES if p not in bounds]
    assert not missing, f"Bounds missing for parameters: {missing}"


def test_fit_config_bounds_are_valid_ranges():
    """Every bound is a two-element list with low <= high."""
    bounds = load_config(str(_FIT_CONFIG))["optimization"]["bounds"]
    for name, (low, high) in bounds.items():
        assert low <= high, f"bounds.{name}: low ({low}) must be ≤ high ({high})"


# Error handling

def test_load_nonexistent_config_raises(tmp_path):
    """Loading a non-existent file raises an error."""
    missing = tmp_path / "does_not_exist.toml"
    with pytest.raises((FileNotFoundError, OSError)):
        load_config(str(missing))
