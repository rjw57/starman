"""
An implementation of the Kalman filter for linear state estimation in the
presence of Gaussian noise.

"""

from __future__ import division, absolute_import, print_function

import numpy as np
import scipy.stats as sps

from .util import as_square_array

class KalmanFilter(object):
    """
    A KalmanFilter maintains an estimate of true state given noisy measurements.

    The filter is initialised to have no state estimates. (Time step "-1" if you
    will.) Before calling :py:meth:`.update`, :py:meth:`.predict` must be called
    at least once.

    The filter represents its state estimates as frozen
    :py:class`scipy.stats.multivariate_normal` instances.

    Args:
        initial_state_estimate (None or scipy.stats.multivariate_normal): The
            initial estimate of the true state used for the first
            :py:meth:`.predict` step. If *None*, *state_length* must be
            specified and the initial state estimate is initialised to zero mean
            and a covariance of the identity matrix muiltiplied by a large
            value. (Specifically the value of
            :py:const:`KalmanFilter.LARGE_COVARIANCE`.)
        process_matrix (array or None): The process matrix
            to use if none is passed to :py:meth:`.predict`.
        process_covariance (array or None): The process noise covariance
            to use if none is passed to :py:meth:`.predict`.
        control_matrix: (array or None): The control matrix to use if none
            is passed to :py:meth:`.predict`.
        state_length (None or int): Must only be specified if
            *initial_state_estimate* is None. In which case, this is used as the
            length of the state vector.

    Raises:
        ValueError: The passed matrices have inconsistent or invalid shapes.

    Attributes:
        prior_state_estimates (list of scipy.stats.multivariate_normal):
            Element *k* is the the *a priori* state estimate for time step *k*.
        posterior_state_estimates (list of scipy.stats.multivariate_normal):
            Element *k* is the the *a posteriori* state estimate for time step
            *k*.
        measurements (list of list of scipy.stats.multivariate_normal):
            Element *k* is a list of :py:class:`scipy.stats.multivariate_normal`
            instances. These are the instances passed to :py:meth:`update` for
            time step *k*.
        process_matrices (list of array): Element *k* is the process matrix used
            by :py:meth:`.predict` at time step *k*.
        process_covariances (list of array): Element *k* is the process
            covariance used by :py:meth:`.predict` at time step *k*.
        measurement_matrices (list of list of array):
            Element *k* is a list of the measurement matrices passed to each
            call to :py:meth:`update` for that time step.
        state_length (int): Number of elements in the state vector.

    """

    #: A large value used as the magnitude of the initial state estimate
    #: covariance when only state vector length is specified.
    LARGE_COVARIANCE = 1e3

    def __init__(self, initial_state_estimate=None, process_matrix=None,
                 process_covariance=None, control_matrix=None,
                 state_length=None):
        if initial_state_estimate is None:
            if state_length is None:
                raise ValueError("state_length must be specified")
            self._initial_state_estimate = sps.multivariate_normal(
                mean=np.zeros(state_length),
                cov=np.eye(state_length) * KalmanFilter.LARGE_COVARIANCE
            )
            self.state_length = state_length
        else:
            if state_length is not None:
                raise ValueError("Only one of state_length and "
                                 "initial_state_estimate should be passed.")
            self._initial_state_estimate = initial_state_estimate
            self.state_length = self._initial_state_estimate.mean.shape[0]

        self._defaults = dict(
            process_matrix=process_matrix,
            process_covariance=process_covariance,
            control_matrix=control_matrix,
        )

        # Initialise prior and posterior estimates
        self.prior_state_estimates = []
        self.posterior_state_estimates = []

        # No measurements just yet
        self.measurements = []
        self.measurement_matrices = []

        # Record of process matrices and covariances passed to predict()
        self.process_matrices = []
        self.process_covariances = []

    def predict(self, control=None, control_matrix=None,
                process_matrix=None, process_covariance=None):
        """
        Predict the next *a priori* state mean and covariance given the last
        posterior. As a special case the first call to this method will
        initialise the posterior and prior estimates from the
        *initial_state_estimate* and *initial_covariance* arguments passed when
        this object was created. In this case the *process_matrix* and
        *process_covariance* arguments are unused but are still recorded in the
        :py:attr:`.process_matrices` and :py:attr:`.process_covariances`
        attributes.

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
            process_matrix = self._defaults['process_matrix']

        if process_covariance is None:
            process_covariance = self._defaults['process_covariance']

        if control_matrix is None:
            control_matrix = self._defaults['control_matrix']

        if len(self.prior_state_estimates) == 0:
            # Special case: first call
            self.prior_state_estimates.append(self._initial_state_estimate)
        else:
            # Usual case
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
            prev_posterior_mean = self.posterior_state_estimates[-1].mean
            prev_posterior_cov = self.posterior_state_estimates[-1].cov

            prior_mean = process_matrix.dot(prev_posterior_mean)
            if control is not None:
                prior_mean += control_matrix.dot(control)

            prior_cov = process_matrix.dot(prev_posterior_cov).dot(
                process_matrix.T) + process_covariance

            self.prior_state_estimates.append(
                sps.multivariate_normal(mean=prior_mean, cov=prior_cov))

        # Record transition matrix
        self.process_matrices.append(process_matrix)
        self.process_covariances.append(process_covariance)

        # Append empty list to measurements for this time step
        self.measurements.append([])
        self.measurement_matrices.append([])

        # Seed posterior estimates with the prior one.
        self.posterior_state_estimates.append(self.prior_state_estimates[-1])

    def update(self, measurement, measurement_matrix):
        """
        After each :py:meth:`predict`, this method may be called repeatedly to
        provide additional measurements for each time step.

        Args:
            measurement (scipy.stats.multivariate_normal): Measurement for this
                time step with specified mean and covariance.
            measurement_matrix (array): Measurement matrix for this measurement.

        """
        # Sanitise input arguments
        measurement_matrix = np.atleast_2d(measurement_matrix)
        expected_meas_mat_shape = (measurement.mean.shape[0], self.state_length)
        if measurement_matrix.shape != expected_meas_mat_shape:
            raise ValueError("Measurement matrix is wrong shape ({}). " \
                    "Expected: {}".format(
                        measurement_matrix.shape, expected_meas_mat_shape))

        # Add measurement list
        self.measurements[-1].append(measurement)
        self.measurement_matrices[-1].append(measurement_matrix)

        # "Prior" in this case means "before we've updated with this
        # measurement".
        prior = self.posterior_state_estimates[-1]

        # Compute Kalman gain
        innovation = measurement.mean - measurement_matrix.dot(prior.mean)
        innovation_cov = measurement_matrix.dot(prior.cov).dot(
            measurement_matrix.T)
        innovation_cov += measurement.cov
        kalman_gain = prior.cov.dot(measurement_matrix.T).dot(
            np.linalg.inv(innovation_cov))

        # Update estimates
        post = self.posterior_state_estimates[-1]
        self.posterior_state_estimates[-1] = sps.multivariate_normal(
            mean=post.mean + kalman_gain.dot(innovation),
            cov=post.cov - kalman_gain.dot(measurement_matrix).dot(prior.cov)
        )

    @property
    def state_count(self):
        """Property returning the number of states/time steps this filter has
        processed. Since the first time step is always 0, the final index will
        always be ``state_count`` - 1.

        """
        return len(self.posterior_state_estimates)

    @property
    def measurement_count(self):
        """Property returning the total number of measurements which have been
        passed to this filter.

        """
        return sum(len(o) for o in self.measurements)
