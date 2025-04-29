"""
.. code-block:: text
    :caption: Usage of the `crm_fit` script

    crm_fit -h

    usage: crm_fit [-h] [-i ITERATION] [-w WORKERS] [-d DATA]
                [-o OUTPUT_FOLDER] [--skip-profiles] [--skip-masks]
                [--skip-param-space] [--skip-distributions]

    Fits the Bacterial Rods model to a system of cells.

    options:
    -h, --help            show this help message and exit
    -i, --iteration ITERATION
                            Use existing output folder instead of
                            creating new one
    -w, --workers WORKERS
                            Number of threads
    -d, --data DATA       Directory containing initial and final
                            snapshots with masks.
    -o, --output-folder OUTPUT_FOLDER
                            Folder to store all output in. If left
                            unspecified, the output folder will be
                            generated via OUTPUT_FOLDER='./out/crm_fit/
                            POTENTIAL_TYPE/ITERATION/' where ITERATION
                            is the next number larger than any already
                            existing one and POTENTIAL_TYPE is obtained
                            from the settings.toml file
    --skip-profiles       Skips Plotting of profiles for parameters
    --skip-masks          Skips Plotting of masks and microscopic
                            images
    --skip-param-space    Skips visualization of parameter space
    --skip-distributions  Skips plotting of distributions
"""

from .plotting import *
from .predict import *
from .main import crm_fit_main
from cr_mech_coli.crm_fit.crm_fit_rs import *
