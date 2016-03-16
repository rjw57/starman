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

Example
-------

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
    Q = diag([1e-1, 1e-1, 1e-2, 1e-2]) ** 2

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

    figure(figsize=(8, 12))

    sca(subplot2grid((4, 1), (0, 0), rowspan=2))
    plot(true_states[:, 0], true_states[:, 1])
    axis("equal")
    grid(True)
    xlabel("x co-ordinate")
    ylabel("y co-ordinate")
    title("True position states")

    sca(subplot2grid((4, 1), (2, 0)))
    plot(true_states[:, 2])
    grid(True)
    ylabel("x velocity")
    title("True x velocity")
    setp(gca().get_xticklabels(), visible=False)

    sca(subplot2grid((4, 1), (3, 0)))
    plot(true_states[:, 3])
    grid(True)
    xlabel("Time")
    ylabel("y velocity")
    title("True y velocity")

    tight_layout()

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
    plot(measurements[:, 0], measurements[:, 1], 'rx:', label="Measured")
    axis("equal")
    grid(True)
    xlabel("x co-ordinate")
    ylabel("y co-ordinate")
    title("True and measured positions")
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
    assert kf.time_step_count == N

    # Check that the filter state dimension is as expected
    assert kf.state_length == STATE_DIM

Now we've run the filter, we can see how it has performed.

.. plot::
    :context: close-figs

    # Stack all the estimated states from the filter into an NxSTATE_DIM array
    estimated_states = vstack(kf.posterior_state_estimates)
    assert estimated_states.shape == (N, STATE_DIM)

    figure(figsize=(8, 12))

    # Plot the position result
    sca(subplot2grid((4, 1), (0, 0), rowspan=2))
    plot(true_states[:, 0], true_states[:, 1], 'b', label="True")
    plot(measurements[:, 0], measurements[:, 1], 'rx:', label="Measured")
    plot(estimated_states[:, 0], estimated_states[:, 1], 'g', label="Estimated")
    axis("equal")
    grid(True)
    xlabel("x co-ordinate")
    ylabel("y co-ordinate")
    title("True, measured and estimated positions")
    legend(loc="best")

    # Plot the velocity result

    sca(subplot2grid((4, 1), (2, 0)))
    plot(true_states[:, 2], 'b', label="True")
    plot(estimated_states[:, 2], 'g', label="Estimated")
    grid(True)
    ylabel("x velocity")
    title("True and estimated x velocity")
    legend(loc="best")
    setp(gca().get_xticklabels(), visible=False)

    sca(subplot2grid((4, 1), (3, 0)))
    plot(true_states[:, 3], 'b', label="True")
    plot(estimated_states[:, 3], 'g', label="Estimated")
    grid(True)
    xlabel("Time")
    ylabel("y velocity")
    title("True and estimated y velocity")

    tight_layout()

We see that the estimates of position and velocity improve over time.

