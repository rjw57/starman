"""
An implementation of the Kalman filter for linear state estimation in the
presence of Gaussian noise.

"""

from __future__ import division, absolute_import, print_function

import numpy as np

from .stats import MultivariateNormal
from .util import as_square_array
from .exc import ParameterError, NoAPrioriStateError

class GaussianStateEstimation(object):
    """
    The :py:class:`.GaussianStateEstimation` class maintains an estimate of the
    true state of a system parametrised via a mean and covariance. At each time
    instance the *a priori* state estimate before measurement and *a posteriori*
    state estimate post-measurement is recorded.

    Each time instant, :math:`k \\in \\mathbb{Z}^+`, has associated with it an *a
    priori* state estimate parametrised by a mean vector, :math:`\\mu_{k|k-1}`,
    and covariance :math:`\\Sigma_{k|k-1}`. These estimates represent the true
    state vector uncertainty with all measurements up to *but not including*
    those at time instant :math:`k`.

    The *a posteriori* state estimate takes into account all measurements up to
    *and including* time step :math:`k`. It is similarly parametrised by a mean,
    :math:`\\mu_{k|k}`, and covariance, :math:`\\Sigma_{k|k}`.

    When initialised, an instance of this class has no *a priori* or *a
    posteriori* measurements. Each call to :py:meth:`.add_prior_estimate` will
    advance to the next time step and set the *a priori* estimate for that
    instance. The first call to :py:meth:`.add_prior_estimate` "advances" to
    time step 0.

    The *a posteriori* state estimate for a time step is modified via the
    :py:meth:`.update_posterior` method. The current time step will not be
    advanced until the next call to :py:meth:`.add_prior_estimate`.

    It follows that there must be at least one call to
    :py:meth:`.add_prior_estimate` before the first call to
    :py:meth:`.update_posterior`.

    State estimates are represented via :py:class:`.MultivariateNormal`
    instances.

    Args:
        state_length (int): The number of elements in the state vector. Used to
            initialise the :py:attr:`.state_length` attribute but also to verify
            the dimensions of arrays passed to methods.

    Raises:
        starman.exc.ParameterError: A value passed at construction was invalid.

    Attributes:
        prior_state_estimates (list of MultivariateNormal):
            Element *k* is the the *a priori* state estimate for time step *k*.
        posterior_state_estimates (list of MultivariateNormal):
            Element *k* is the the *a posteriori* state estimate for time step
            *k*.
        measurements (list of list of MultivariateNormal):
            Element *k* is a list of :py:class:`.MultivariateNormal`
            instances. These are the instances passed to :py:meth:`update` for
            time step *k*.

    """
    def __init__(self, state_length):
        # Record state vector length
        self.state_length = int(state_length)

        # Initialise prior and posterior estimates
        self.prior_state_estimates = []
        self.posterior_state_estimates = []

        # No measurements just yet
        self.measurements = []

    def add_prior_estimate(self, estimate):
        """
        Provide a new *a priori* estimate of state for the next time step. This
        has the effect of advancing the current time step. This method must be
        called at least once before :py:meth:`.update_posterior`.

        Args:
            estimate (MultivariateNormal): state estimate for next time step.

        Raises:
            starman.exc.ParameterError: the state estimate had incorrect
                dimension.

        """
        if estimate.mean.shape[0] != self.state_length:
            raise ParameterError('State vector must have length {}'.format(
                self.state_length))

        # Add estimate to prior state estimates
        self.prior_state_estimates.append(estimate)

        # Append empty list to measurements for this time step
        self.measurements.append([])

        # Seed posterior estimates with the a priori one.
        self.posterior_state_estimates.append(self.prior_state_estimates[-1])

    def update_posterior(self, measurement, estimate):
        """
        After each :py:meth:`add_prior_estimate`, this method may be called
        repeatedly to provide additional measurements and associated *a
        posteriori* estimates for each time step.

        Args:
            measurement (MultivariateNormal): Measurement for this
                time step with specified mean and covariance.
            estimate (MultivariateNormal): *A posteriori* state estimate after
                this measurement.

        Raises:
            starman.exc.ParameterError: the state estimate had incorrect
            starman.exc.NoAPrioriStateError: the :py:meth:`.add_prior_estimate`
                method has not yet been called.

        """
        if self.time_step is None:
            raise NoAPrioriStateError('No a priori state estimates')
        if estimate.mean.shape[0] != self.state_length:
            raise ParameterError('State vector must have length {}'.format(
                self.state_length))

        # Add measurement to list
        self.measurements[-1].append(measurement)

        # Update posterior
        self.posterior_state_estimates[-1] = estimate

    def truncate(self, new_count):
        """Truncate the estimator as if only *new_count*
        :py:meth:`.add_prior_estimate`, steps had been performed. If *new_count*
        is greater than :py:attr:`.state_count` then this function is a no-op.

        Measurements, state estimates, process matrices and process noises which
        are truncated are discarded.

        Args:
            new_count (int): Number of states to retain.

        """
        self.posterior_state_estimates = \
            self.posterior_state_estimates[:new_count]
        self.prior_state_estimates = self.prior_state_estimates[:new_count]
        self.measurements = self.measurements[:new_count]

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

    @property
    def time_step(self):
        """Property giving the current time step which calls to
        :py:meth:`.update_posterior` will update. If
        :py:meth:`.add_prior_estimate` has not yet been called, this property
        will be *None*.

        """
        n_states = self.state_count
        return n_states - 1 if n_states > 0 else None

class KalmanFilter(GaussianStateEstimation):
    """
    A KalmanFilter maintains an estimate of true state given noisy measurements.
    It is a subclass of :py:class:`.GaussianStateEstimation`.

    Args:
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
        starman.exc.ParameterError: The passed matrices have inconsistent or
            invalid shapes.

    Attributes:
        prior_state_estimates (list of MultivariateNormal):
            Element *k* is the the *a priori* state estimate for time step *k*.
        posterior_state_estimates (list of MultivariateNormal):
            Element *k* is the the *a posteriori* state estimate for time step
            *k*.
        measurements (list of list of MultivariateNormal):
            Element *k* is a list of :py:class:`.MultivariateNormal`
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
    def __init__(self, state_length,
                 process_matrix=None, process_covariance=None,
                 control_matrix=None):
        GaussianStateEstimation.__init__(self, state_length=state_length)

        self._defaults = dict(
            process_matrix=process_matrix,
            process_covariance=process_covariance,
            control_matrix=control_matrix,
        )

        # No measurements just yet
        self.measurement_matrices = []

        # Record of process matrices and covariances passed to predict()
        self.process_matrices = []
        self.process_covariances = []

    def clone(self):
        """Return a new :py:class:`.KalmanFilter` instance which is a shallow
        clone of this one. By "shallow", although the lists of measurements,
        etc, are cloned, the :py:class:`.MultivariateNormal` instances within
        them are not. Since :py:meth:`.predict` and :py:meth:`.update` do not
        modify the elements of these lists, it is safe to run two cloned filters
        in parallel as long as one does not directly modify the states.

        Returns:
            (KalmanFilter): A new :py:class:`KalmanFilter` instance.

        """
        new_f = KalmanFilter(state_length=self.state_length)
        new_f._defaults = self._defaults # pylint:disable=protected-access
        new_f.state_length = self.state_length
        new_f.prior_state_estimates = list(self.prior_state_estimates)
        new_f.posterior_state_estimates = list(self.posterior_state_estimates)
        new_f.measurements = list(self.measurements)
        new_f.process_matrices = list(self.process_matrices)
        new_f.process_covariances = list(self.process_covariances)

        return new_f

    def set_initial_state(self, estimate, process_matrix=None,
                          process_covariance=None):
        """
        Adds an *a priori* state estimate directly to the filter. This method
        should be used to initialise the filter with the initial state estimate.

        Args:
            estimate (MultivariateNormal): The state estimate for the new time
                step.
            process_matrix (array or None): The process matrix to
                use for this time step. If None, use the default value.
            process_covariance (array or None): The process
                covariance to use for this time step. If None, use the default
                value.

        Raises:
            starman.exc.ParameterError: if the filter has already been
                initialised.

        """
        if self.time_step is not None:
            raise ParameterError('Initial state already set')
        self._add_prior_estimate(estimate, process_matrix, process_covariance)

    def _add_prior_estimate(self, estimate, process_matrix=None,
                            process_covariance=None):
        self.add_prior_estimate(estimate)

        # Sanitise arguments
        if process_matrix is None:
            process_matrix = self._defaults['process_matrix']

        if process_covariance is None:
            process_covariance = self._defaults['process_covariance']

        if process_matrix is not None:
            process_matrix = as_square_array(process_matrix)

        if process_covariance is not None:
            process_covariance = as_square_array(process_covariance)

        if process_matrix is not None and process_covariance is not None and \
                process_matrix.shape[0] != process_covariance.shape[0]:
            raise ValueError("Process matrix and noise have incompatible " \
                             "shapes: {} vs {}".format(
                                 process_matrix.shape, process_covariance.shape))

        # Add new measurement matrix list
        self.measurement_matrices.append([])

        # Record transition matrix
        self.process_matrices.append(process_matrix)
        self.process_covariances.append(process_covariance)

    def predict(self, control=None, control_matrix=None,
                process_matrix=None, process_covariance=None):
        """
        Predict the next *a priori* state mean and covariance given the last
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
            process_matrix = self._defaults['process_matrix']

        if process_covariance is None:
            process_covariance = self._defaults['process_covariance']

        if control_matrix is None:
            control_matrix = self._defaults['control_matrix']

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

        # Add prediction.
        self._add_prior_estimate(
            MultivariateNormal(mean=prior_mean, cov=prior_cov),
            process_matrix, process_covariance)

    def update(self, measurement, measurement_matrix):
        """
        After each :py:meth:`predict`, this method may be called repeatedly to
        provide additional measurements for each time step.

        Args:
            measurement (MultivariateNormal): Measurement for this
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
        self.update_posterior(measurement, MultivariateNormal(
            mean=post.mean + kalman_gain.dot(innovation),
            cov=post.cov - kalman_gain.dot(measurement_matrix).dot(prior.cov)
        ))

        # Add measurement matrix to list
        self.measurement_matrices[-1].append(measurement_matrix)

    def truncate(self, new_count):
        GaussianStateEstimation.truncate(self, new_count)
        self.process_matrices = self.process_matrices[:new_count]
        self.process_covariances = self.process_covariances[:new_count]
        self.measurement_matrices = self.measurement_matrices[:new_count]
