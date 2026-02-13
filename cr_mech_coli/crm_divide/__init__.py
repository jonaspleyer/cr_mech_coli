"""
This script is specifically designed for only one particular example contained in
`data/crm_divide/0001/`.
Therefore it will not work with any other dataset provided.
It is used to estimate the parameters of our model and provide profiles for them.
More specifically, it estimates the parameters under division of agents.

.. code-block:: text
    :caption: Usage of the `crm_amir` script

    usage: crm_divide [-h] [-i ITERATION] [--output-dir OUTPUT_DIR] [--skip-profiles]
                    [--skip-time-evolution] [--skip-snapshots] [--skip-timings]
                    [--skip-mask-adjustment] [--only-mask-adjustment]
                    [--skip-distribution] [-w WORKERS]
                    [--profiles-maxiter PROFILES_MAXITER]
                    [--profiles-samples PROFILES_SAMPLES]
                    [--profiles-optim-method PROFILES_OPTIM_METHOD]
                    [--profiles-lhs-sample-size PROFILES_LHS_SAMPLE_SIZE]
                    [--profiles-lhs-maxiter PROFILES_LHS_MAXITER] [--data-dir DATA_DIR]
                    {DE,LHS} ...

    Fits the Bacterial Rods model to a system of cells.

    positional arguments:
    {DE,LHS}
        DE                  Use the differential_evolution algorithm for optimization
        LHS                 Use the Latin-Hypercube Sampling with some local minimization
                            for optimization

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
    --skip-distribution   Skip plotting of the distribution of growth rates
    -w, --workers WORKERS
                            Number of threads to utilize
    --profiles-maxiter PROFILES_MAXITER
    --profiles-samples PROFILES_SAMPLES
    --profiles-optim-method PROFILES_OPTIM_METHOD
    --profiles-lhs-sample-size PROFILES_LHS_SAMPLE_SIZE
    --profiles-lhs-maxiter PROFILES_LHS_MAXITER
    --data-dir DATA_DIR

"""

from cr_mech_coli.crm_divide.crm_divide_rs import *
from .main import *
