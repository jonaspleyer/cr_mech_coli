Synthetic Microscope Image Generation (``crm_gen``)
----------------------------------------------------

``crm_gen`` provides tools for generating realistic synthetic microscope images of rod-shaped bacteria. It combines the ``cr_mech_coli`` simulation
framework with a modular image synthesis pipeline, enabling the creation of
labelled training data for cell segmentation models and the validation of
imaging workflows against real microscope data.

The module is available as both a command-line tool and a Python library:

.. code-block:: bash

    crm_gen --config my_config.toml run                     # full simulation + rendering pipeline
    crm_gen --config my_config.toml clone img.tif mask.tif  # clone a real microscope image
    crm_gen --config my_config.toml fit                     # fit parameters to real images


Module Structure
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 80
   :width: 100%

   * - Submodule
     - Description
   * - :doc:`pipeline <pipeline>`
     - Orchestrates the full simulation-to-image pipeline; main programmatic entry point.
   * - :doc:`scene <scene>`
     - Composites a single frame: rendered cells on background with microscope effects applied.
   * - :doc:`background <background>`
     - Generates synthetic phase contrast background textures.
   * - :doc:`bacteria <bacteria>`
     - Assigns per-cell brightness based on cell age or real image intensities.
   * - :doc:`filters <filters>`
     - Applies PSF blur, phase contrast halos, Poisson shot noise, and Gaussian readout noise.
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
