Introduction
============

The starman library provides implementation of algorithms commonly used to
estimate state and to track targets over time in the presence of noisy
measurements.

Those wanting to dive in and see what is supported may take a look at the
:doc:`reference` for all the gory details.

Features
--------

Starman provides a Kalman filter implementation which can be used to track the
hidden true state of a linear system over time given zero or more noisy
measurements for each time step.

A Rauch-Tung-Striebel smoother (RTS) implementation is provided which, when
combined with the Kalman filter, can produce very smooth estimates of state.
Unlike the Kalman filter which provides an estimate of state for each time step
based only on measurements up until that time step, the RTS smoother requires
all measurements to have been recorded.

For associating multiple measurements per frame to multiple targets, an
implementation of the Scott and Longuet-Higgins feature association algorithm is
provided. This algorithm can be used to "join the dots" when tracking multiple
targets.

Why "starman"?
--------------

Starman implements the Kalman filter. The Kalman filter was used for trajectory
estimation in the Apollo spaceflight programme. Starman is thus a blend of
"star", signifying space, and "Kalman". That and "kalman" was already taken as a
package name on the PyPI.
