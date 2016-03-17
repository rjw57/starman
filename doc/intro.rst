Introduction
============

The starman library implements state estimation algorithms for tracking the
hidden state of a system given noisy observations. Take a look at the
:doc:`reference` for all the gory details.

Features
--------

A flexible Kalman filter implementation is provided via the
:py:class:`starman.KalmanFilter` class. Use a Kalman filter to estimate the
hidden state of a linear system in the presence of Gaussian noise.

Kalman filters are "online"; they provide estimates of the current state of the
system given past measurements. If you have all the measurements available to
you, you can use an "offline" method which estimates the state at each time
instant given all past *and* future measurements. A Rauch-Tung-Striebel (RTS)
smoother uses the Kalman filter outputs and recursively computes this "all data"
estimate. The :py:func:`rts_smooth` function provides a RTS implementation.

See :doc:`kalman` for more details and example code.

Why "starman"?
--------------

Starman implements the Kalman filter. The Kalman filter was used for trajectory
estimation in the Apollo spaceflight programme. Starman is thus a blend of
"star", signifying space, and "Kalman". That and "kalman" was already taken as a
package name on the PyPI.
