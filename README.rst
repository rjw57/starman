.. image:: https://travis-ci.org/rjw57/starman.svg?branch=master
    :target: https://travis-ci.org/rjw57/starman

.. image:: https://coveralls.io/repos/github/rjw57/starman/badge.svg?branch=master
    :target: https://coveralls.io/github/rjw57/starman?branch=master

.. image:: http://readthedocs.org/projects/starman/badge/?version=latest
    :target: http://starman.readthedocs.org/en/latest/?badge=latest
    :alt: Documentation Status

Starman: State-estimation and tracking for Python
=================================================

Starman is a library which implements algorithms of use when trying to track the
true state of one or more systems over time in the presence of noisy
observations.

Full `documentation <http://starman.readthedocs.org/en/latest/>`_ is available
on readthedocs.

Features
--------

Currently starman supports the following algorithms:

* Kalman filtering for state estimation.
* Rauch-Tung-Striebel smoothing for the Kalman filter.
* Scott and Longuet-Higgins feature association for matching measurments to
  tracked states.

Copyright and licensing
-----------------------

See the `LICENCE.txt <LICENSE.txt>`_ file in the repository root for details.
tl;dr: MIT-style.

Why "starman"?
--------------

Starman implements the Kalman filter. The Kalman filter was used for trajectory
estimation in the Apollo spaceflight programme. Starman is thus a blend of
"star", signifying space, and "Kalman". That and "kalman" was already taken as a
package name on the PyPI.

