"""
This module contains functionality to configure and run simulations.

.. list-table:: Define Agent Properties
    :header-rows: 0
    :widths: 40 60

    * - :class:`AgentSettings`
      - A template for defining an agent.
    * - :class:`RodMechanicsSettings`
      - Defines properties of to the `RodMechanics struct
        <https://cellular-raza.com/docs/cellular_raza_building_blocks/struct.RodMechanics.html>`_.
    * - :class:`MorsePotentialF32`
      - Define interaction properties of the agent.


.. list-table:: Running a Simulation
    :header-rows: 0
    :widths: 40 60

    * - :class:`Configuration`
      - Bundles all information required for a simulation.
    * - :func:`run_simulation`
      - Executes the simulation and returns a :class:`CellContainer`.
"""

from .cr_mech_coli_rs import (
    run_simulation,
    run_simulation_with_agents,
    AgentSettings,
    RodAgent,
    RodMechanicsSettings,
    Configuration,
    sort_cellular_identifiers,
    CellIdentifier,
    MorsePotentialF32,
)
