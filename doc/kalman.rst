.. default-role:: math

Kalman filter
=============

In this section we discuss the ``starman`` support for the `Kalman filter
<https://en.wikipedia.org/wiki/Kalman_filter>`_.

Mathematical overview
---------------------

Let's first refresh the goal the Kalman filter and its formulation. The Kalman
filter attempts to update an estimate of the "true" state of a system given
noisy measurements of the state. The state is assumed to evolve in a linear way
and measurements are assumed to be linear functions of the state. Specifically,
it is assumed that the "true" state at time `k+1` is a function of the "true"
state at time `k`:

.. math::

    x_{k+1} = F_k x_k + B_k u_k + w_k

where `w_k` is a sample from a zero-mean Gaussian process with covariance `Q_k`.
We term `Q_k` the *process covariance*.  The matrix `F_k` is termed the
*state-transition matrix* and determines how the state evolves. The matrix `B_k`
is the *control matrix* and determines the contribution to the state of the
control input, `u_k`.

At time instant `k` we may have zero or more *measurements* of the state. Each
measurement, `z_k` is assumed to be a linear function of the state:

.. math::

    z_k = H_k x_k + v_k

where `H_k` is termed the *measurement matrix* and `v_k` is a sample from a
zero-mean Gaussian process with covariance `R_k`. We term `R_k` the *measurement
covariance*.

The Kalman filter maintains for time instant, `k`, an *a priori* estimate of
state, `\hat{x}_{k|k-1}` covariance of this estimate, `P_{k|k-1}`. The initial
values of these parameters are given when the Kalman filter is created. The
filter also maintains an *a posteriori* estimate of state, `\hat{x}_{k|k}`, and
covariance, `P_{k|k}`. This is updated for each measurement, `z_k`.

*A Priori* Prediction
`````````````````````

At time `k` we are given a state transition matrix, `F_k`, and estimate of the
*process noise*, `Q_k`. Our *a priori* estimates are then given by:

.. math::
    \hat{x}_{k|k-1} = F_k \hat{x}_{k-1|k-1} + B_k u_k,
    \quad
    P_{k|k-1} = F_k P_{k-1|k-1} F_k^T + Q_k.

Innovation
``````````

At time `k` we are given a matrix, `H_k`, which specifies how a given
measurement is derived from the state and some estimate of the measurement noise
covariance, `R_k`.  We may now compute the innovation, `y_k`, of the measurement
from the predicted measurement and our expected innovation covariance, `S_k`:

.. math::

    y_k = z_k - H_k \hat{x}_{k|k-1}, \quad S_k = H_k P_{k|k-1} H_k^T + R_k.

Update
``````

We now update the state estimate with the measurement via the so-called *Kalman
gain*, `K_k`:

.. math::

    K_k = P_{k|k-1} H_k^T S_k^{-1}.

Merging is straightforward. Note that if we have no measurement, our *a
posteriori* estimate reduces to the *a priori* one:

.. math::

    \hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k y_k, \quad P_{k|k} = P_{k|k-1} - K_k
    H_k P_{k|k-1}.

.. _const-vel-kalman:

A simple example: the constant velocity model
---------------------------------------------

The Kalman filter is implemented in ``starman`` via the
:py:class:`starman.KalmanFilter` class. This section provides an example of use.

Generating the true states
``````````````````````````

We will implement a simple 2D state estimation problem using the constant
velocity model. The state transition matrix is constant throughout the model:

.. plot::
    :context: reset

    # Import numpy and matplotlib functions into global namespace
    from matplotlib.pylab import *

    # Our state is x-position, y-position, x-velocity and y-velocity.
    # The state evolves by adding the corresponding velocities to the
    # x- and y-positions.
    F = array([
        [1, 0, 1, 0], # x <- x + vx
        [0, 1, 0, 1], # y <- y + vy
        [0, 0, 1, 0], # vx is constant
        [0, 0, 0, 1], # vy is constant
    ])

    # Specify the length of the state vector
    STATE_DIM = 4

Let's generate some sample data by determining the process noise covariance:

.. plot::
    :context: close-figs

    from numpy.random import multivariate_normal

    # Specify the process noise covariance
    Q = diag([5e-2, 5e-2, 1e-2, 1e-2]) ** 2
    # How many states should we generate?
    N = 100

    # Generate some "true" states
    initial_state = zeros(STATE_DIM)
    true_states = [initial_state]
    for _ in range(N-1):
        # Next state is determined by last state...
        next_state = F.dot(true_states[-1])
        # ...with added process noise
        next_state += multivariate_normal(mean=zeros(STATE_DIM), cov=Q)
        # Record the state
        true_states.append(next_state)
    assert len(true_states) == N

    # Stack all the true states into a single NxSTATE_DIM array
    true_states = vstack(true_states)
    assert true_states.shape == (N, STATE_DIM)

We can plot the true states we've just generated:

.. plot::
    :context: close-figs

    import matplotlib.gridspec as gridspec

    # Convenience function to set up our plotting axes
    def create_axes():
        gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])
        ax_xy = subplot(gs[0, :])
        ax_vx = subplot(gs[1, 0])
        ax_vy = subplot(gs[1, 1])

        ax_xy.set_xlabel("X co-ordinate")
        ax_xy.set_ylabel("Y co-ordinate")
        ax_xy.grid(True)
        ax_vx.set_ylabel("X velocity")
        ax_vx.set_xlabel("Time step")
        ax_vx.grid(True)
        ax_vy.set_ylabel("Y velocity")
        ax_vy.set_xlabel("Time step")
        ax_vy.grid(True)

        return ax_xy, ax_vx, ax_vy

    figure(figsize=(8, 12))
    ax_xy, ax_vx, ax_vy = create_axes()
    tight_layout()

    sca(ax_xy)
    plot(true_states[:, 0], true_states[:, 1])
    axis("equal")

    sca(ax_vx)
    plot(true_states[:, 2])

    sca(ax_vy)
    plot(true_states[:, 3])


Generating measurements
```````````````````````

We will use a measurement model where the velocity is a "hidden" state and we
can only directly measure position. We'll also specify a measurement error
covariance.

.. plot::
    :context: close-figs

    # We only measure position
    H = array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ])

    # And we measure with some error
    R = diag([0.1, 0.1]) ** 2

    # Specify the measurement vector length
    MEAS_DIM = 2

From the measurement matrix and measurement error we can generate noisy
measurements from the true states.

.. plot::
    :context: close-figs

    # Generate measurements
    measurements = []

    for state in true_states:
        # Measure state...
        z = H.dot(state)

        # ...with added measurement noise
        z += multivariate_normal(mean=zeros(MEAS_DIM), cov=R)

        # Record measurement
        measurements.append(z)

    # Stack the measurements into an NxMEAS_DIM array
    measurements = vstack(measurements)
    assert measurements.shape == (N, MEAS_DIM)

Let's plot the measurements overlaid on the true states.

.. plot::
    :context: close-figs

    plot(true_states[:, 0], true_states[:, 1], label="True")
    plot(measurements[:, 0], measurements[:, 1], 'rx:', label="Measured", alpha=0.5)
    axis("equal"); grid(True); xlabel("x co-ordinate"); ylabel("y co-ordinate")
    legend(loc="best")

Using the Kalman filter
```````````````````````

We can create an instance of the :py:class:`starman.KalmanFilter` to filter our
noisy measurements.

.. plot::
    :context: close-figs

    from starman import KalmanFilter

    # Our initial state estimate has very high covariances
    initial_state_estimate = zeros(STATE_DIM)
    initial_covariance = 1e10 * diag(ones(STATE_DIM))

    # Create a kalman filter with known process and measurement matrices and
    # known covariances.
    kf = KalmanFilter(
        initial_state_estimate=initial_state_estimate,
        initial_covariance=initial_covariance,
        process_matrix=F, process_covariance=Q,
        measurement_matrix=H, measurement_covariance=R
    )

    # For each time step
    for k, z in enumerate(measurements):
        # There's no point predicting for the first time step
        if k != 0:
            kf.predict()

        # Update filter with measurement
        kf.update(z)

    # Check that filter length is as expected
    assert kf.state_count == N

    # Check that the filter state dimension is as expected
    assert kf.state_length == STATE_DIM

Now we've run the filter, we can see how it has performed.

.. plot::
    :context: close-figs

    # Stack all the estimated states from the filter into an NxSTATE_DIM array
    estimated_states = vstack(kf.posterior_state_estimates)
    assert estimated_states.shape == (N, STATE_DIM)

    # Stack all the estimated covariances into an NxSTATE_DIMxSTATE_DIM array.
    estimated_covs = vstack(c[newaxis, ...] for c in kf.posterior_state_covariances)
    assert estimated_covs.shape == (N, STATE_DIM, STATE_DIM)

    # Convenience function to plot a value with variances. Shades the n sigma
    # region.
    def plot_vars(x, y, y_vars, n=3.0, **kwargs):
        y_sigma = sqrt(y_vars)
        fill_between(x, y - n*y_sigma, y + n*y_sigma, **kwargs)

    # Get array of timesteps
    ks = np.arange(estimated_states.shape[0])

    figure(figsize=(8, 12))
    ax_xy, ax_vx, ax_vy = create_axes()
    tight_layout()

    sca(ax_xy)
    plot(true_states[:, 0], true_states[:, 1], 'b', label="True")
    plot(measurements[:, 0], measurements[:, 1], 'rx:', label="Measured", alpha=0.5)
    plot(estimated_states[:, 0], estimated_states[:, 1], 'g', label="Estimated")
    axis("equal"); legend(loc="best")

    sca(ax_vx)
    plot(true_states[:, 2], 'b', label="True")
    plot(estimated_states[:, 2], 'g', label="Estimated")
    gca().autoscale(False)
    plot_vars(ks, estimated_states[:, 2], estimated_covs[:, 2, 2],
              color='g', alpha=0.25, zorder=-1)

    sca(ax_vy)
    plot(true_states[:, 3], 'b', label="True")
    plot(estimated_states[:, 3], 'g', label="Estimated")
    gca().autoscale(False)
    plot_vars(ks, estimated_states[:, 3], estimated_covs[:, 3, 3],
              color='g', alpha=0.25, zorder=-1)

We see that the estimates of position and velocity improve over time.

Rauch-Tung-Striebel smoothing
-----------------------------

The `Rauch-Tung-Striebel
<https://en.wikipedia.org/wiki/Kalman_filter#Rauch.E2.80.93Tung.E2.80.93Striebel>`_
(RTS) smoother provides a method of computing the "all data" *a posteriori*
estimate of states (as opposed to the "all previous data" estimate). Assuming
there are `n` time points in the filter, then the RTS computes the *a
posteriori* state estimate at time `k` after all the data for `n` time steps are
known, `\hat{x}_{k|n}`, and corresponding covariance, `P_{k|n}`, recursively:

.. math::

    \hat{x}_{k|n} = \hat{x}_{k|k} + C_k ( \hat{x}_{k+1|n} - \hat{x}_{k+1|k} ),
    \quad P_{k|n} = P_{k|k} + C_k ( P_{k+1|n} - P_{k+1|k} ) C_k^T

with `C_k = P_{k|k} F^T_{k+1} P_{k+1|k}^{-1}`.

The RTS smoother is an example of an "offline" algorithm in that the estimated
state for time step `k` depends on having seen *all* of the measurements rather
than just the measurements up until time `k`.

Using RTS smoothing
```````````````````

We'll start by assuming that the steps in :ref:`const-vel-kalman` have been
performed. Namely that we have some true states in ``true_states``, measurements
in ``measurements`` and a :py:class:`starman.KalmanFilter` instance in ``kf``.

Following on from that example, we can use the :py:func:`starman.rts_smooth`
function to compute the smoothed state estimates given all of the data.

.. plot::
    :context: close-figs

    from starman import rts_smooth

    # Compute the smoothed states given all of the data
    rts_states, rts_covs = rts_smooth(kf)
    assert rts_states.shape == (N, STATE_DIM)
    assert rts_covs.shape == (N, STATE_DIM, STATE_DIM)

    # Plot the result
    figure(figsize=(8, 12))
    ax_xy, ax_vx, ax_vy = create_axes()
    tight_layout()

    sca(ax_xy)
    plot(true_states[:, 0], true_states[:, 1], 'b', label="True")
    plot(measurements[:, 0], measurements[:, 1], 'rx:', label="Measured", alpha=0.5)
    plot(estimated_states[:, 0], estimated_states[:, 1], 'g', label="Kalman filter")
    plot(rts_states[:, 0], rts_states[:, 1], 'm', label="RTS")
    axis("equal"); legend(loc="best")

    sca(ax_vx)
    plot(true_states[:, 2], 'b', label="True")
    plot(estimated_states[:, 2], 'g', label="Estimated")
    plot(rts_states[:, 2], 'm', label="RTS")
    gca().autoscale(False)
    plot_vars(ks, rts_states[:, 2], rts_covs[:, 2, 2],
              color='m', alpha=0.25, zorder=-1)

    sca(ax_vy)
    plot(true_states[:, 3], 'b', label="True")
    plot(estimated_states[:, 3], 'g', label="Estimated")
    plot(rts_states[:, 3], 'm', label="RTS")
    gca().autoscale(False)
    plot_vars(ks, rts_states[:, 3], rts_covs[:, 3, 3],
              color='m', alpha=0.25, zorder=-1)

We can see how the RTS smoothed states are far smoother than the forward
estimated states.

