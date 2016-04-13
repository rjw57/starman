#!/usr/bin/env python3
"""
Show an example of constant-velocity model state estimation via the Kalman
filter.

Usage:
    kalman_filtering.py [<file>]

If specified, the resulting plot is written to <file> instead of being shown.
"""

import docopt
from matplotlib.pylab import *
from numpy.random import multivariate_normal as sample_mvn
from starman import KalmanFilter, MultivariateNormal

opts = docopt.docopt(__doc__)

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

# Specify the process noise covariance
Q = diag([1e-3, 1e-3, 1e-2, 1e-2]) ** 2

# How many states should we generate?
N = 100

# Generate some "true" states
initial_state = zeros(STATE_DIM)
true_states = [initial_state]

for _ in range(N-1):
    # Next state is determined by last state...
    next_state = F.dot(true_states[-1])

    # ...with added process noise
    next_state += sample_mvn(mean=zeros(STATE_DIM), cov=Q)

    # Record the state
    true_states.append(next_state)

assert len(true_states) == N

# Stack all the true states into a single NxSTATE_DIM array
true_states = vstack(true_states)
assert true_states.shape == (N, STATE_DIM)

# We only measure position
H = array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
])

# And we measure with some error
R = diag([0.2, 0.2]) ** 2

# Specify the measurement vector length
MEAS_DIM = 2

# Generate measurements
measurements = []

for state in true_states:
    # Measure state...
    z = H.dot(state)

    # ...with added measurement noise
    z += sample_mvn(mean=zeros(MEAS_DIM), cov=R)

    # Record measurement
    measurements.append(z)

# Stack the measurements into an NxMEAS_DIM array
measurements = vstack(measurements)
assert measurements.shape == (N, MEAS_DIM)


# Create a kalman filter with known process and measurement matrices and
# known covariances.
kf = KalmanFilter(state_length=STATE_DIM,
                  process_matrix=F, process_covariance=Q)

# For each time step
for k, z in enumerate(measurements):
    # Predict state for this time step
    if k != 0:
        kf.predict()

    # Update filter with measurement
    kf.update(MultivariateNormal(z, R), H)

# Check that filter length is as expected
assert kf.state_count == N

# Check that the filter state dimension is as expected
assert kf.state_length == STATE_DIM

# Stack all the estimated states from the filter into an NxSTATE_DIM array
estimated_states = vstack([d.mean for d in kf.posterior_state_estimates])
assert estimated_states.shape == (N, STATE_DIM)

# Plot the position result
figure(figsize=(15, 6))

sca(subplot2grid((2, 2), (0, 0), rowspan=2))
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

sca(subplot2grid((2, 2), (0, 1)))
plot(true_states[:, 2], 'b', label="True")
plot(estimated_states[:, 2], 'g', label="Estimated")
grid(True)
ylabel("x velocity")
title("True and estimated x velocity")
legend(loc="best")
setp(gca().get_xticklabels(), visible=False)

sca(subplot2grid((2, 2), (1, 1)))
plot(true_states[:, 3], 'b', label="True")
plot(estimated_states[:, 3], 'g', label="Estimated")
grid(True)
xlabel("Time")
ylabel("y velocity")
title("True and estimated y velocity")

if opts['<file>']:
    tight_layout()
    savefig(opts['<file>'])
else:
    show()


