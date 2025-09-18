"""
.. code-block:: text
    :caption: Usage of the `crm_divide` script

    crm_divide -h

    usage: crm_divide [-h] [-i ITERATION] [--output-dir OUTPUT_DIR] [--skip-profiles] [--skip-time-evolution]
                    [--skip-snapshots] [--skip-timings] [--skip-mask-adjustment] [--only-mask-adjustment]
                    [-w WORKERS]

    Fits the Bacterial Rods model to a system of cells.

    options:
    -h, --help            show this help message and exit
    -i, --iteration ITERATION
                            Use existing output folder instead of creating new one
    --output-dir OUTPUT_DIR
                            Directory where to store results
    --skip-profiles       Skip plotting of profiles
    --skip-time-evolution
                            Skip plotting of the time evolution of costs
    --skip-snapshots      Skip plotting of snapshots and masks
    --skip-timings        Skip plotting of the timings
    --skip-mask-adjustment
                            Skip plotting of the adjusted masks
    --only-mask-adjustment
                            Only plot adjusted masks
    -w, --workers WORKERS
                            Number of threads to utilize

"""

from .main import crm_divide_main
from cr_mech_coli.crm_divide.crm_divide_rs import *
