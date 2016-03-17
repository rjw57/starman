import numpy as np
from numpy.random import multivariate_normal as sample_mvn

from starman import KalmanFilter, rts_smooth, MultivariateNormal

# Our state is x-position, y-position, x-velocity and y-velocity.
# The state evolves by adding the corresponding velocities to the
# x- and y-positions.
F = np.array([
    [1, 0, 1, 0], # x <- x + vx
    [0, 1, 0, 1], # y <- y + vy
    [0, 0, 1, 0], # vx is constant
    [0, 0, 0, 1], # vy is constant
])

# Specify the length of the state vector
STATE_DIM = 4

# Specify the process noise covariance
Q = np.diag([1e-2, 1e-2, 1e-2, 1e-2]) ** 2

# How many states should we generate?
N = 100

# We only measure position
H = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
])

# And we measure with some error
R = np.diag([0.1, 0.1]) ** 2

# Specify the measurement vector length
MEAS_DIM = 2

def generate_true_states(count=N):
    # Generate some "true" states
    initial_state = np.zeros(STATE_DIM)
    true_states = [initial_state]

    for _ in range(count-1):
        # Next state is determined by last state...
        next_state = F.dot(true_states[-1])

        # ...with added process noise
        next_state += sample_mvn(mean=np.zeros(STATE_DIM), cov=Q)

        # Record the state
        true_states.append(next_state)

    # Stack all the true states into a single NxSTATE_DIM array
    true_states = np.vstack(true_states)
    assert true_states.shape == (count, STATE_DIM)

    return true_states

def generate_measurements(true_states):
    # Generate measurements
    measurements = []

    for state in true_states:
        # Measure state...
        z = H.dot(state)

        # ...with added measurement noise
        z += sample_mvn(mean=np.zeros(MEAS_DIM), cov=R)

        # Record measurement
        measurements.append(z)

    # Stack the measurements into an NxMEAS_DIM array
    measurements = np.vstack(measurements)
    assert measurements.shape == (N, MEAS_DIM)

    return measurements

def create_filter(true_states, measurements):
    # Create a kalman filter with known process and measurement matrices and
    # known covariances.
    kf = KalmanFilter(state_length=STATE_DIM,
                      process_matrix=F, process_covariance=Q)

    # For each time step
    for k, z in enumerate(measurements):
        # Predict
        kf.predict()

        # Update filter with measurement
        kf.update(MultivariateNormal(mean=z, cov=R), H)

    # Check that filter length is as expected
    assert kf.state_count == N

    # Check that the filter state dimension is as expected
    assert kf.state_length == STATE_DIM

    return kf

def test_kalman_basic():
    true_states = generate_true_states()
    measurements = generate_measurements(true_states)
    kf = create_filter(true_states, measurements)

    # Stack all the estimated states from the filter into an NxSTATE_DIM array
    estimated_states = np.vstack([e.mean for e in kf.posterior_state_estimates])
    assert estimated_states.shape == (N, STATE_DIM)

    # It is vanishingly unlikely that we're wrong by 5 sigma.
    for est, true in zip(kf.posterior_state_estimates, true_states):
        delta = est.mean - true
        dist = delta.dot(np.linalg.inv(est.cov)).dot(delta)
        assert dist < 5*5

def test_rts_smooth():
    true_states = generate_true_states()
    measurements = generate_measurements(true_states)
    kf = create_filter(true_states, measurements)

    # Perform RTS smoothing
    estimates = rts_smooth(kf)

    # It is vanishingly unlikely that we're wrong by 5 sigma.
    for est, true in zip(estimates, true_states):
        delta = est.mean - true
        dist = delta.dot(np.linalg.inv(est.cov)).dot(delta)
        assert dist < 5*5
