Fitting Methods
===============

- Talk about how to measure differences between masks

.. subfigure:: AB
    :layout-sm: A|B
    :gap: 8px
    :subcaptions: below
    :class-grid: outline

    .. image:: _static/fitting-methods/progressions-1.png
    .. image:: _static/fitting-methods/progressions-2.png

    Snapshots at `t=20min` and `t=40min`.
    All cells have undergone a division event.

.. subfigure:: AB
    :layout-sm: A|B
    :gap: 8px
    :subcaptions: below
    :class-grid: outline

    .. image:: _static/fitting-methods/progressions-3.png
    .. image:: _static/fitting-methods/progressions-4.png

    Calculations of differences between the images.
    The first image purely calculates the differing area while the second approach also takes into
    account if cells are related and weighs this specific overlapping area less.

.. figure:: _static/fitting-methods/penalty-time-flow.png

   We performed penalty calculations for simulation snapshots in lock-step with two different
   techniques.
   Due to the linear growth of the cells, we expect a constant amount of difference reported by our
   implemented methods.
   We can clearly see that cell-division introduces undesirable spikes between individual
   time-steps.
   Relations between daughter and parent-cells have been weighted with a penalty of :math:`p=0`.

New text here
