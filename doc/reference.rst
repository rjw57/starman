Programmer's Reference
======================

Below is a description of the public API of starman separated by functionality.
In-depth discussion how to use the API can be found in the appropriate sections
of the documentation.

State estimation
----------------

.. autoclass:: starman.KalmanFilter
    :members:
    :member-order: bysource

.. autofunction:: starman.rts_smooth

Feature association
-------------------

.. autofunction:: starman.slh_associate

Representation of state estimates
---------------------------------

.. autoclass:: starman.MultivariateNormal
    :members:

Helper functions for linear systems
-----------------------------------

.. automodule:: starman.linearsystem
    :members:
