Synthetic Microscope Image Generation (``crm_gen``)
----------------------------------------------------

``crm_gen`` provides tools for generating realistic synthetic microscope
images of rod-shaped bacteria.
It combines the ``cr_mech_coli`` simulation framework with a modular image
synthesis pipeline, enabling the creation of labelled training data for cell
segmentation models and the validation of imaging workflows against real
microscopic data.

The module is available as both a command-line tool and a Python library:

.. code-block:: bash

    crm_gen run                                           # full simulation + rendering pipeline
    crm_gen run --config my_gen.toml                      # custom generation config
    crm_gen clone img.tif mask.tif                        # clone a real microscope image
    crm_gen clone img.tif mask.tif --config my_gen.toml
    crm_gen fit path/to/real/images/                      # fit parameters to real images
    crm_gen fit path/to/real/images/ --config my_fit.toml

``run`` and ``clone`` use a *generation config* (imaging and simulation
parameters). ``fit`` uses a separate *fit config* (optimisation
hyperparameters only: ``maxiter``, ``popsize``, etc.). The imaging
parameters themselves are the *output* of the fit.


Module Structure
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 80
   :width: 100%

   * - Submodule
     - Description
   * - :doc:`pipeline <pipeline>`
     - Runs the full cell growth simulation and applies synthetic
       microscope effects to each frame via ``scene``; main programmatic entry point.
   * - :doc:`scene <scene>`
     - Composites a single synthetic frame using ``background``,
       ``bacteria``, and ``filters``.
   * - :doc:`background <background>`
     - Generates synthetic phase contrast background textures.
   * - :doc:`bacteria <bacteria>`
     - Assigns per-cell brightness based on cell age or real image intensities.
   * - :doc:`filters <filters>`
     - Simulates optical effects (Point Spread Function (PSF) blur and phase
       contrast halos) and sensor noise (Poisson shot noise and Gaussian
       readout noise).
   * - :doc:`metrics <metrics>`
     - Computes SSIM, PSNR, and histogram distance between synthetic and real images.
   * - :doc:`optimization <optimization>`
     - Optimises imaging parameters to match real images via differential evolution.
   * - :doc:`visualization <visualization>`
     - Generates diagnostic plots for inspecting optimisation results.
   * - :doc:`configuration <configuration>`
     - Loads and manages TOML configuration files and default parameter constants.

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :hidden:

    pipeline <pipeline>
    scene <scene>
    background <background>
    bacteria <bacteria>
    filters <filters>
    metrics <metrics>
    optimization <optimization>
    visualization <visualization>
    configuration <configuration>
