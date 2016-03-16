"""
An implementation of the Kalman filter for linear state estimation in the
presence of Gaussian noise.

"""

from __future__ import division, absolute_import, print_function

import numpy as np
from .util import as_square_array

class KalmanFilter(object):
    # pylint:disable=too-many-instance-attributes
    """
    A KalmanFilter maintains an estimate of true state given noisy measurements.

    Args:
        initial_state_estimate (array): The initial *a priori* state estimate
        initial_covariance (array): The initial *a priori* state estimate
            covariance.
        process_matrix (array or None): The process matrix
            to use if none is passed to :py:meth:`.predict`.
        process_covariance (array or None): The process noise covariance
            to use if none is passed to :py:meth:`.predict`.
        control_matrix: (array or None): The control matrix to use if none
            is passed to :py:meth:`.predict`.
        measurement_matrix (array or None): The measurement matrix
            to use if none is passed to :py:meth:`.update`.
        measurement_covariance (array or None): The measurement noise
            covariance to use if none is passed to :py:meth:`.update`.

    Raises:
        ValueError: The passed matrices have inconsistent or invalid shapes.

    Attributes:
        prior_state_estimates (list of array): A list of *a priori* state
            estimates.
        prior_state_covariances (list of array): A list of *a priori* state
            covariances.
        posterior_state_estimates (list of array): A list of *a posteriori*
            state estimates.
        posterior_state_covariances (list of array): A list of *a posteriori*
            state covariances.
        process_matrices (list of array or None): The process matrices used for
            corresponding predict steps. The initial entry will be *None* since
            there is no state transition for the first state.
        process_covariances (list of array or None): The process covariances
            used for corresponding predict steps. The initial entry will be
            *None* since there is no state transition for the first state.
        measurements (list of list of tuple): This list holds the measurements
            passed to :py:meth:`update` for each time step. Each time step has a
            list of (measurement, measurement matrix, measurement covariance)
            triples. Time steps with no measurements have an empty list.

    """

    def __init__(self, initial_state_estimate, initial_covariance,
                 process_matrix=None, process_covariance=None,
                 control_matrix=None,
                 measurement_matrix=None, measurement_covariance=None):
        # pylint:disable=too-many-arguments

        self.process_matrix = process_matrix
        self.process_covariance = process_covariance
        self.control_matrix = control_matrix
        self.measurement_matrix = measurement_matrix
        self.measurement_covariance = measurement_covariance

        # Initialise prior and posterior estimates
        self.prior_state_estimates = [np.copy(initial_state_estimate)]
        self.prior_state_covariances = [as_square_array(np.copy(initial_covariance))]
        self.posterior_state_estimates = [self.prior_state_estimates[-1]]
        self.posterior_state_covariances = [self.prior_state_covariances[-1]]

        # Used by RTS smoother first transition matrix corresponding to initial
        # mean and covariance is meaningless. Add placeholder of None.
        self.process_matrices = [None]
        self.process_covariances = [None]

        # The measurements list holds records of measurements associated with
        # the filter at time k. If no measurements are recorded, this will be an
        # empty list. Otherwise it is a list of measurement, measurement_matrix,
        # measurement_covariance triples.
        self.measurements = [[]]

        # Index at which the last measurement was associated
        self.last_measurement_time_step = None

    def predict(self, control=None, control_matrix=None,
                process_matrix=None, process_covariance=None):
        """Predict the next *a priori* state mean and covariance given the last
        posterior.

        Args:
            control (array or None): If specified, the control input for this
                predict step.
            control_matrix (array or None): If specified, the control matrix to
                use for this time step.
            process_matrix (array or None): If specified, the process matrix to
                use for this time step.
            process_covariance (array or None): If specified, the process
                covariance to use for this time step.

        """
        # Sanitise arguments
        if process_matrix is None:
            process_matrix = self.process_matrix

        if process_covariance is None:
            process_covariance = self.process_covariance

        if control_matrix is None:
            control_matrix = self.control_matrix

        process_matrix = as_square_array(process_matrix)
        process_covariance = as_square_array(process_covariance)
        if process_matrix.shape[0] != process_covariance.shape[0]:
            raise ValueError("Process matrix and noise have incompatible " \
                "shapes: {} vs {}".format(
                    process_matrix.shape, process_covariance.shape))

        if control_matrix is not None:
            control_matrix = np.atleast_2d(control_matrix)
        if control is not None:
            control = np.atleast_1d(control)

        # Update state mean and covariance
        prior_state = process_matrix.dot(self.posterior_state_estimates[-1])
        if control is not None:
            prior_state += control_matrix.dot(control)
        self.prior_state_estimates.append(prior_state)
        self.prior_state_covariances.append(
            process_matrix.dot(self.posterior_state_covariances[-1]).dot(
                process_matrix.T) +
            process_covariance
        )

        # Record transition matrix
        self.process_matrices.append(process_matrix)
        self.process_covariances.append(process_covariance)

        # Append empty list to measurements for this time step
        self.measurements.append([])

        # Seed posterior estimates with *copies* of the prior ones
        self.posterior_state_estimates.append(
            np.copy(self.prior_state_estimates[-1]))
        self.posterior_state_covariances.append(
            np.copy(self.prior_state_covariances[-1]))

    def update(self, measurement,
               measurement_matrix=None, measurement_covariance=None):
        """After each :py:meth:`predict`, this method may be called repeatedly
        to provide additional measurements for each time step.

        Args:
            measurement (array): Measurement for this time step
            measurement_matrix (array or None): Measurement matrix for this
                measurement
            measurement_covariance (array or None): Measurement covariance for
                this measurement.

        """
        # Sanitise input arguments
        if measurement_matrix is None:
            measurement_matrix = self.measurement_matrix
        if measurement_covariance is None:
            measurement_covariance = self.measurement_covariance

        measurement_matrix = np.atleast_2d(measurement_matrix)
        measurement_covariance = as_square_array(measurement_covariance)
        measurement = np.atleast_1d(measurement)

        expected_obs_mat_shape = (measurement_covariance.shape[0], self.state_length)
        if measurement_matrix.shape != expected_obs_mat_shape:
            raise ValueError("Observation matrix is wrong shape ({}). " \
                    "Expected: {}".format(
                        measurement_matrix.shape, expected_obs_mat_shape))
        if measurement.shape != (measurement_covariance.shape[0],):
            raise ValueError("Observation is wrong shape ({}). " \
                    "Expected: {}".format(
                        measurement.shape, (measurement_covariance.shape[0],)))

        # Add measurement triple to list
        self.measurements[-1].append(
            (measurement, measurement_matrix, measurement_covariance))
        self.last_measurement_time_step = len(self.measurements) - 1

        # "Prior" in this case estimates "before we've updated with this
        # measurement".
        prior_mean = np.copy(self.posterior_state_estimates[-1])
        prior_covariance = np.copy(self.posterior_state_covariances[-1])

        # Can compute innovation covariance & Kalman gain without an measurement
        innovation = measurement - measurement_matrix.dot(prior_mean)
        innovation_cov = measurement_matrix.dot(prior_covariance).dot(
            measurement_matrix.T)
        innovation_cov += measurement_covariance
        kalman_gain = prior_covariance.dot(measurement_matrix.T).dot(
            np.linalg.inv(innovation_cov))

        # Update estimates
        self.posterior_state_estimates[-1] += kalman_gain.dot(innovation)
        self.posterior_state_covariances[-1] -= kalman_gain.dot(measurement_matrix).dot(
            prior_covariance)

    @property
    def time_step_count(self):
        """Property returning the number of time steps for this filter."""
        return len(self.posterior_state_estimates)

    @property
    def state_length(self):
        """Property returning the number of elements in the state vector."""
        return self.prior_state_estimates[-1].shape[0]

    @property
    def measurement_count(self):
        """Property returning the total number of measurements which have been
        passed to this filter.

        """
        return sum(len(o) for o in self.measurements)
