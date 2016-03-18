Feature Association
===================

When estimating the state of a single system, techniques such as Kalman
filtering can be extremely useful. Real situations often have several systems
acting independently each of which can generate a measurement. Sometimes it is
clear which measurement has arisen from which system. Sometimes it is not.
*Feature association* is the process of associating actual measurements with
predicted measurements from a set of tracked systems.

Starman supports the following feature association algorithms:

.. toctree::
    :maxdepth: 1

    slh
