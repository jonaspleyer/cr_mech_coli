Image-Generation
================

Comparing Microscopic Images
----------------------------

.. subfigure:: ABC
    :layout-sm: A|B|C
    :gap: 8px
    :subcaptions: below
    :class-grid: outline

    .. image:: _static/image-generation/Coli3.jpg
        :alt: Electron Microscopy
    .. image:: _static/image-generation/image001133-cropped.png
        :alt: Light Microscopy (1)
    .. image:: _static/image-generation/E_choli_Gram-cropped.JPG
        :alt: Light Microscopy (2)

    There exist a great variety of microscopes which generate differing images
    :cite:`WikimediaPicturesEColi`.
    For our interests, images taken by Electron Microscopes :cite:`Knoll1932` can not be taken into
    account since the imaging process kills the bacteria and thus no dynamics can be captured.

- Microscopic images
- Surface of the Cell

List of Imaging Efects
----------------------

.. list-table:: Effects Introduced by Light Microscopes
    :widths: 30 30 40
    :header-rows: 1

    * - Effect
      - CG Equivalent
      - Description
    * - Sensory Noise
      - Uniform noise at each pixel
      - Sensors are noisy?
    * - "Smudges"
      - Distortion Layers
      - Persistent across single time-series
    * - Lensing
      -
      -
    * - Angle of Lighting
      -
      -
    * - Intensity
      -
      -
    * - Indirect Lighting/Refraction
      -
      -
    * - Optical Aberration
      -
      -

3D Rendering
------------

- Which effects can we capture with that?


Calculating Masks
-----------------

- parallel projection
- maybe switch to something less calculation-instense (such as matplotlib)
    - can we still trust that the generated image and the masks match?
