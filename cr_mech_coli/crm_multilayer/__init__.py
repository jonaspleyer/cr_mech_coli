"""
This script is designed to calculate and visualize simulation results which lead to transitions
from 2D cell sheets to 3D colonies.

.. warning::
    This script is under active development.
    It is not ready to be used yet.


.. code-block:: text
    :caption: Usage of the `crm_multilayer` script

    usage: crm_multilayer [-h] [-w WORKERS] {run,plot} ...

    Run Simulations to analyze Multilayer-behaviour of Rod-Shaped Bacteria.

    positional arguments:
    {run,plot}
        run                 Run simulation for specified parameters
        plot                Perform plotting actions

    options:
    -h, --help            show this help message and exit
    -w, --workers WORKERS
                            Total number of threads to be used
"""

from cr_mech_coli.crm_multilayer.crm_multilayer_rs import *

from .main import crm_multilayer_main
from .runner import *
from .plotting import *
