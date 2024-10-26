Model
#####

Bacteria exist in many physical shapes such as spheroidal, rod-shaped or spiral
:cite:`Zapun2008,Young2006`.

To model the spatial mechanics of elongated bacteria :cite:`Billaudeau2017`, we represent them as a
collection of auxiliary vertices :math:`\{\vec{v}_i\}` which are connected by springs in
ascending order.
Furthermore, we assume that the cells are flexible described by their stiffness property.
A force :math:`\vec{F}` interacting between cellular agents determines the radius (thickness) of the
rods and an attractive component can model adhesion between cells.

Mechanics
---------

In principle we can assign individual lengths :math:`\{l_i\}` and strengths
:math:`\{\gamma\}_i` to each spring.
The internal force acting on vertex :math:`\vec{v}_i` can be divided into 2 contributions coming
from the 2 springs pulling on it.
In the case when :math:`i=0,N_\text{vertices}`, this is reduced to only one internal component.
We denote with :math:`\vec{c}_{i}` the connection between two vertices

.. math:: \vec{c}_i = \vec{v}_{i}-\vec{v}_{i-1}

and can write down the resulting force

.. math::
    \vec{F}_{i,\text{springs}} =
        &-\gamma_i\left(1 - \frac{l_i}{\left|\vec{c}_i\right|}\right)\vec{c}_i\\
        &+ \gamma_{i+1}\left(1 - \frac{l_{i+1}}{\left|\vec{c}_{i+1}\right|}\right)\vec{c}_{i+1}
    :label: force-springs

In addition to springs between individual vertices :math:`\vec{v}_i`, we assume that each angle at
a vertex between two other is subject to a stiffening force.
Assuming that :math:`\alpha_i` is the angle between the connections and
:math:`\vec{d}_i=\vec{c}_i/|\vec{c}_i|` is the normalized connection,
we can write down the forces acting on vertices :math:`\vec{v}_i,\vec{v}_{i-1},\vec{v}_{i+1}`

.. math::
    \vec{F}_{i,\text{stiffness}} &= \eta_i\left(\pi-\alpha_i\right)
        \frac{\vec{d}_i - \vec{d}_{i+1}}{|\vec{d}_i-\vec{d}_{i+1}|}\\
    \vec{F}_{i-1,\text{stiffness}} &= -\frac{1}{2}\vec{F}_{i,\text{stiffness}}\\
    \vec{F}_{i+1,\text{stiffness}} &= -\frac{1}{2}\vec{F}_{i,\text{stiffness}}
    :label: force-stiffness

where :math:`\eta_i` is the angle stiffness at vertex :math:`\vec{v}_i`.
We can see that the stiffening force does not move the overall center of the cell in space.
The total force is the sum of external and interal forces.

.. math::
    \vec{F}_{i,\text{total}} = \vec{F}_{i,\text{springs}}+ \vec{F}_{i,\text{stiffness}} + \vec{F}_{i,\text{external}}
   :label: force-total

and are integrated via

.. math::
    \partial_t^2 \vec{x} &= \partial\vec{x} + D\vec{\xi}\\
    \partial_t\vec{x} &= \vec{F}_\text{total}
    :label: equations-of-motion

where :math:`D` is the diffusion constant and  :math:`\vec{\xi}` is the wiener process (compare with
brownian motion such as given by the
`Brownian3D <https://cellular-raza.com/docs/cellular_raza_building_blocks/struct.Brownian3D.html>`_
struct of `cellular-raza <https://cellular-raza.com>`_.

.. _ TODO insert table with all parameters

Interaction
-----------

When calculating forces acting between the cells, we can use a simplified model to circumvent the
numerically expensive integration over the complete length of the rod.
Given a vertex :math:`\vec{v}_i` on one cell, we calculate the closest point :math:`\vec{p}` on the polygonal
line given by the vertices :math:`\{\vec{w}_j\}` of the interacting cell.
Furthermore we determine the value :math:`q\in[0,1]` such that

.. math:: \vec{p} = (1-q)\vec{w}_j + q\vec{w}_{j+1}
   :label: connection

for some specific :math:`j`.
The force is then calculated between the points :math:`\vec{v}_i` and :math:`\vec{p}_i` and acts on
the vertex :math:`\vec{w}_i,\vec{w}_{i+1}` with relative strength :math:`(1-q)` and :math:`q`.

.. math:: \vec{F}_{i,\text{External}} = \vec{F}(\vec{v}_i,\vec{p})
    :label: force-external

.. _ TODO MORSEPOTENTIAL

.. _ TODO insert table with all parameters

Cycle
-----

To simulate proliferation, we introduce a growth term for the spring lengths :math:`l_i`

.. math:: \partial_t l_i = \mu
    :label: growth-ode

which will increase the length of the cell indefenitely unless we introduce a
`division event <https://cellular-raza.com/internals/concepts/cell/cycle>`_.
We define a threshold (in our case double of the original length) for the total length of the
cell at which it divides.
To construct a new cell, we cannot simply copy the existing one twice, but we also need to adjust
internal parameters in the process.
The following actions need to be taken for the old and new agent.

.. _ TODO

1. Assign a new growth rate (pick randomly from uniform distribution in :math:`[0.8\mu_0,1.2\mu_0]`
   where :math:`\mu_0` is some fixed value)
2. Assign new positions
    1. Calculate new spring lengths
        :math:`\tilde{l}_i = l_i\left(\frac{1}{2} - \frac{r}{\sum\limits_i l_i}\right)`
    2. Calculate middle of old cell
        :math:`\vec{m} = \frac{1}{N_\text{vertices}}\sum\limits_i\vec{v}_i`
    3. Calculate positions of new vertices :math:`\vec{w}_i`
        .. math::
            q_i &= \frac{i}{N_\text{vertices}}\\
            \vec{w}_{i,\text{new},\pm} &= (1-q_i)\vec{m} + q_i(\vec{v}_{\pm\text{start}} - \vec{m})
