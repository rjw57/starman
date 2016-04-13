"""
An implementation of the Kalman filter for linear state estimation in the
presence of Gaussian noise.

"""

from __future__ import division, absolute_import, print_function

import numpy as np

from .stats import MultivariateNormal
from .util import as_square_array
from .exc import ParameterError

#: A large value used as the magnitude of the initial state estimate
#: covariance when only state vector length is specified.
LARGE_COVARIANCE = 1e8

def form_initial_state(initial_state_estimate, state_length):
    """Implements the logic of GaussianStateEstimation for forming the initial
    state estimate from *initial_state_estimate* and *state_length* parameters.

    Returns:
        A tuple giving the initial state estimate and state length.

    """
    if initial_state_estimate is None:
        if state_length is None:
            raise ParameterError("state_length must be specified")
        initial_state_estimate = MultivariateNormal(
            mean=np.zeros(state_length),
            cov=np.eye(state_length) * LARGE_COVARIANCE
        )
    else:
        if state_length is not None:
            raise ParameterError("Only one of state_length and "
                                 "initial_state_estimate should be passed.")
        state_length = initial_state_estimate.mean.shape[0]

    return initial_state_estimate, state_length

class GaussianStateEstimation(object):
    """
    A :py:class:`.GaussianStateEstimation` maintains an estimate of the true
    state of a system parametrised via a mean and covariance. At each time
    instance the *a priori* state estimate before measurement and *a posteriori*
    state estimate post-measurement is recorded.

    Time is advanced by calling the :py:meth:`.new_prediction` method. The *a
    posteriori* state estimate for a time step is modified via the
    :py:meth:`.update_posterior` method.

    The object is initialised to have no state estimates. (Time step "-1" if you
    will.) Before calling :py:meth:`.update_posterior`,
    :py:meth:`.new_prediction` or :py:meth:`reset` must be called at least once.

    State estimates are represented via :py:class:`.MultivariateNormal`
    instances.

    Args:
        initial_state_estimate (None or MultivariateNormal): The
            initial estimate of the true state used for the first
            :py:meth:`.predict` step. If *None*, *state_length* must be
            specified and the initial state estimate is initialised to zero mean
            and a covariance of the identity matrix muiltiplied by a large
            value. (Specifically the value of
            :py:data:`.LARGE_COVARIANCE`.)
        state_length (None or int): Must only be specified if
            *initial_state_estimate* is None. In which case, this is used as the
            length of the state vector.

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
        state_length (int): Number of elements in the state vector.

    """
    def __init__(self, initial_state_estimate=None, state_length=None):
        self._initial_state_estimate, self.state_length = \
            form_initial_state(initial_state_estimate, state_length)

        # Initialise prior and posterior estimates
        self.prior_state_estimates = []
        self.posterior_state_estimates = []

        # No measurements just yet
        self.measurements = []

    def reset(self):
        """
        Resets state estimates to timestep 0. The *a priori* and *a posteriori*
        estimates are initialised from the initial state estimates passed at
        construction time.

        """
        self.prior_state_estimates = [self._initial_state_estimate]
        self.posterior_state_estimates = [self._initial_state_estimate]
        self.measurements = [[]]

    def new_prediction(self, estimate):
        """
        Provide a new *a priori* estimate of state for the next time step. This
        has the effect of advancing the current time step. This method must be
        called at least once before :py:meth:`.update_posterior`.

        Args:
            estimate (MultivariateNormal): state estimate for next time step.

        """
        # Add estimate to prior state estimates
        self.prior_state_estimates.append(estimate)

        # Append empty list to measurements for this time step
        self.measurements.append([])

        # Seed posterior estimates with the a priori one.
        self.posterior_state_estimates.append(self.prior_state_estimates[-1])

    def update_posterior(self, measurement, estimate):
        """
        After each :py:meth:`new_prediction`, this method may be called
        repeatedly to provide additional measurements and associated *a
        posteriori* estimates for each time step.

        Args:
            measurement (MultivariateNormal): Measurement for this
                time step with specified mean and covariance.
            estimate (MultivariateNormal): *A posteriori* state estimate after
                this measurement.

        """
        # Add measurement to list
        self.measurements[-1].append(measurement)

        # Update posterior
        self.posterior_state_estimates[-1] = estimate

    def truncate(self, new_count):
        """Truncate the estimator as if only *new_count*
        :py:meth:`.new_prediction`, steps had been performed. If *new_count* is
        greater than :py:attr:`.state_count` then this function is a no-op.

        Measurements, state estimates, process matrices and process noises which
        are truncated are discarded.

        Args:
            new_count (int): Number of states to retain.

        """
        self.posterior_state_estimates = self.posterior_state_estimates[:new_count]
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

class KalmanFilter(GaussianStateEstimation):
    """
    A KalmanFilter maintains an estimate of true state given noisy measurements.
    It is a subclass of :py:class:`.GaussianStateEstimation`.

    The filter is initialised to have no state estimates. (Time step "-1" if you
    will.) Before calling :py:meth:`.update`, :py:meth:`.predict` must be called
    at least once.

    The filter represents its state estimates as frozen
    :py:class:`.MultivariateNormal` instances.

    Args:
        initial_state_estimate (None or MultivariateNormal): The
            initial estimate of the true state used for the first
            :py:meth:`.predict` step. If *None*, *state_length* must be
            specified and the initial state estimate is initialised to zero mean
            and a covariance of the identity matrix muiltiplied by a large
            value. (Specifically the value of
            :py:data:`.LARGE_COVARIANCE`.)
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
    def __init__(self, initial_state_estimate=None, process_matrix=None,
                 process_covariance=None, control_matrix=None,
                 state_length=None):
        GaussianStateEstimation.__init__(
            self, initial_state_estimate=initial_state_estimate,
            state_length=state_length)

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
        new_f = KalmanFilter(
            initial_state_estimate=self._initial_state_estimate)
        new_f._defaults = self._defaults # pylint:disable=protected-access
        new_f.state_length = self.state_length
        new_f.prior_state_estimates = list(self.prior_state_estimates)
        new_f.posterior_state_estimates = list(self.posterior_state_estimates)
        new_f.measurements = list(self.measurements)
        new_f.process_matrices = list(self.process_matrices)
        new_f.process_covariances = list(self.process_covariances)

        return new_f

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

        # Add new measurement matrix list
        self.measurement_matrices.append([])

        # Usual case
        process_matrix = as_square_array(process_matrix)
        process_covariance = as_square_array(process_covariance)
        if process_matrix.shape[0] != process_covariance.shape[0]:
            raise ValueError("Process matrix and noise have incompatible " \
                "shapes: {} vs {}".format(
                    process_matrix.shape, process_covariance.shape))

        # Record transition matrix
        self.process_matrices.append(process_matrix)
        self.process_covariances.append(process_covariance)

        # Special case: first call resets to timestep 0
        if len(self.prior_state_estimates) == 0:
            self.reset()
            return

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

        # Add prediction
        self.new_prediction(MultivariateNormal(mean=prior_mean, cov=prior_cov))

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
