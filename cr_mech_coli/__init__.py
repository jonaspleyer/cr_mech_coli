"""
This package performs numerical simulations of rod-shaped bacterial cells.
It can also visualize the results of these simulations and perform parameter estimations.

Performing a simulation and storing the corresponding images is straightforward.
All settings needed to run a single simulation are contained in the :class:`.Configuration` class.
The routine is executed via the :func:`.run_simulation` function.

>>> import cr_mech_coli as crm
>>> config = crm.Configuration()
>>> config.n_agents = 8
>>> sim_result = crm.run_simulation(config)

To adjust visualization settings, we provide the :class:`.RenderSettings` class.

>>> render_settings = RenderSettings()
>>> render_settings.noise = 30
>>> crm.store_all_images(config, sim_result, save_dir="out")

.. note::
    This package is based on the `f32 <https://doc.rust-lang.org/std/primitive.f32.html>`_ floating
    point type.
    All numerical calculations performed by `cellular_raza <https://cellular-raza.com>`_ are done
    in this format.
    However the same can not be guaranteed for calculations involving any of the python packages.
"""

from .datatypes import *
from .fitting import *
from .simulation import *
from .imaging import *

from .crm_fit import *
from .crm_multilayer import *
