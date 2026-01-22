"""
The `crm_amir` script is designed to specifically estimate the parameters of the bending experiment
performed by Amir et. al :cite:`Amir2014_2`.
With our developed methods we are able to properly estimate the parameters of our model.
More details are contained in our upcoming publication.

.. code-block:: text
    :caption: Usage of the `crm_amir` script

    usage: crm_amir [-h] [-w WORKERS] [--maxiter MAXITER] [--popsize POPSIZE] [--skip-polish]
                [--maxiter-profiles MAXITER_PROFILES] [--optim-tol-profiles OPTIM_TOL_PROFILES]
                [--optim-atol-profiles OPTIM_ATOL_PROFILES] [--init INIT]
                [--popsize-profiles POPSIZE_PROFILES] [--skip-polish-profiles]
                [--samples-profiles SAMPLES_PROFILES]

    options:
    -h, --help            show this help message and exit
    -w, --workers WORKERS
                            Number of threads (default: -1)
    --maxiter MAXITER     Maximum iterations of the optimization routine (default: 350)
    --popsize POPSIZE     Population Size of the optimization routine (default: 30)
    --skip-polish         Skips polishing the result of the differential evolution algorithm (default:
                            False)
    --maxiter-profiles MAXITER_PROFILES
                            See MAXITER (default: 350)
    --optim-tol-profiles OPTIM_TOL_PROFILES
                            Relative Tolerance for optimization within profiles (default: 0.0001)
    --optim-atol-profiles OPTIM_ATOL_PROFILES
                            Absolute Tolerance for optimization within profiles (default: 0.01)
    --init INIT           Initialization method for the sampling of parameters. (default: latinhypercube)
    --popsize-profiles POPSIZE_PROFILES
                            See POPSIZE (default: 30)
    --skip-polish-profiles
                            See POLISH (default: False)
    --samples-profiles SAMPLES_PROFILES
                            Number of sample points for profile likelihood plots (default: 100)
"""

from cr_mech_coli.crm_amir.crm_amir_rs import *

from .main import crm_amir_main
