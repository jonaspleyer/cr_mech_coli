crm_gen
-------

``crm_gen`` generates realistic synthetic phase contrast microscope images of
bacteria. It combines the ``cr_mech_coli`` simulation framework with a modular
image synthesis pipeline and exposes three subcommands.

.. code-block:: bash

    crm_gen run   [--config path/to/gen_config.toml]
    crm_gen clone img.tif mask.tif [--config path/to/gen_config.toml]
    crm_gen fit   path/to/real/images/ [--config path/to/fit_config.toml]

``run`` and ``clone`` use a *generation config*; ``fit`` uses a separate
*fit config*. Default configs are located in
``cr_mech_coli/crm_gen/configs/``.

----

**crm_gen run**

Runs a bacteria growth simulation and applies synthetic microscope effects to
each rendered frame. Each simulation produces paired output (a synthetic
phase contrast image and a pixel-accurate segmentation mask), making this the
primary tool for generating labelled training data for deep learning
segmentation models.

.. code-block:: bash

    crm_gen run
    crm_gen run --config my_gen.toml

----

**crm_gen clone**

Creates a synthetic version of a real microscope image by extracting cell
positions from a segmentation mask and re-rendering them with synthetic
optical effects.

.. code-block:: bash

    crm_gen clone image.tif mask.tif
    crm_gen clone image.tif mask.tif --output ./out --config my_gen.toml

Options:

- ``--output`` / ``-o`` — output directory (default: ``./synthetic_output``)
- ``--n-vertices`` — vertices per cell, overrides config (default: 8)
- ``--seed`` — random seed, overrides config

----

**crm_gen fit**

Optimises the 7 synthetic imaging parameters to match a directory of real
microscope images using differential evolution. The imaging parameters are the
*output* of the fit; the fit config contains only optimisation hyperparameters
and search bounds.

.. code-block:: bash

    crm_gen fit path/to/real/images/
    crm_gen fit path/to/real/images/ --config my_fit.toml

The fit config (``configs/default_fit_config.toml``) contains:

- ``[optimization]`` — hyperparameters (``maxiter``, ``popsize``, ``workers``, etc.)
- ``[optimization.bounds]`` — search bounds ``[min, max]`` for each imaging parameter
- ``[optimization.metric_weights]`` — loss weights (SSIM, PSNR, histogram distance)
- ``[optimization.region_weights]`` — foreground vs. background loss weighting

