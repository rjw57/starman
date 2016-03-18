State estimation
################

It is often easy enough to write down equations determining the dynamics of a
system: how its state varies over time. Given a system at time `k` we can
predict what state it will be in at time `k+1`. We can also take measurements on
the system at time `k+1`. The process of fusing zero or measurements of a system
with predictions of its state is called *state estimation*.

.. include:: partial_kalman.rst
