Fitting-Methods
===============

Extracting Positions from Masks
-------------------------------

.. subfigure:: ABC
    :layout-sm: A|B|C
    :gap: 8px
    :subcaptions: below
    :class-grid: outline

    .. image:: _static/fitting-methods/extract_positions-002000.png
    .. image:: _static/fitting-methods/extract_positions-006000.png
    .. image:: _static/fitting-methods/extract_positions-010000.png

    Snapshots at `40min`, `80min` and `120min`
    All images show masks which have been generated by a simulation run.
    The positions associated to the individual cell agents which have been estimated by our fitting
    process are displayed with dark crosses while the segments which actually constitute the cell
    are displayed as a contiguous bright line.

.. figure:: _static/fitting-methods/displacement-calculations.png

    To quantify the effectiveness of our fitting algorithm, we compare the average rod lengths of
    the known simulated agent and teh eestiamted positions with each other.
    Furthermore, we calculated the average difference per vertex between the simulated position and
    the estimated one.
    We can clearly see that the fitting method slightly underestimates the total rod length.
    This can be attributed to the Skelezonization algorithm :cite:`Lee1994` which truncates the ends
    of the point set more.
    As time increases, our fitting method becomes less accurate.
    However, overall slopes and division events are still captured correctly.
    This behaviour is due non-trivial geometries of the cell which makes it harder to properly
    estimate the approximating polygon.

- Calculate individual mask segments for cells
- Skeletonize cell-mask :cite:`Lee1994`
- Sort points along major axis
- Approximate polygon :cite:`wiki:Ramer–Douglas–Peucker_algorithm`
- Calculate evenly-spaced segments along polygon

Constucting a Cost Function
---------------------------

- Talk about how to measure differences between masks

.. subfigure:: AB
    :layout-sm: A|B
    :gap: 8px
    :subcaptions: below
    :class-grid: outline

    .. image:: _static/fitting-methods/progressions-1.png
    .. image:: _static/fitting-methods/progressions-2.png

    Snapshots at `t=40min` and `t=60min`.
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

   We performed penalty calculations for successive simulation snapshots with two different
   techniques.
   Due to the overall exponential growth of the ensemble, we also expect an exponential difference
   (its derivative) reported by our implemented methods as time increases.
   We can clearly see that cell-division introduces undesirable spikes between individual
   time-steps when not accounting for cell-lineage.
   When weighing differences between daughter and parent-cells with a penalty of :math:`p=0`, these
   enormous spikes are regularized.
