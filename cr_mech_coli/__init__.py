"""
This package performs numerical simulations of rod-shaped bacterial cells.
It can also visualize the results of these simulations and provides methods to perform parameter
estimations.

Running a simulation and storing the corresponding images is straightforward.
All settings needed to run a single simulation are contained in the :class:`.Configuration` class.
The routine is executed via the :func:`.run_simulation` function and returns a
:class:`.CellContainer` which contains all information of the cells at every stored time interval.

.. code-block:: python

    import cr_mech_coli as crm

    # Contains settings regarding simulation domain, time increments etc.
    config = crm.Configuration()

    # Use predefined values for agents
    agent_settings = crm.AgentSettings()

    # Automatically generate agents
    agents = crm.generate_agents(
        4,
        agent_settings,
        config
    )

    # Run simulation and return container
    cell_container = crm.run_simulation_with_agents(config, agents)

    # Plot individual results
    crm.store_all_images(cell_container, config.domain_size)

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
from .plotting import *
