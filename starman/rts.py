"""
Rauch-Tung-Striebel smoother for Kalman filters.

"""
# pylint:disable=redefined-builtin
from builtins import range

import numpy as np

from .stats import MultivariateNormal

def rts_smooth(kalman_filter, state_count=None):
    """
    Compute the Rauch-Tung-Striebel smoothed state estimates and estimate
    covariances for a Kalman filter.

    Args:
        kalman_filter (KalmanFilter): Filter whose smoothed states should be
            returned
        state_count (int or None): Number of smoothed states to return.
            If None, use ``kalman_filter.state_count``.

    Returns:
        (list of MultivariateNormal): List of multivariate normal distributions.
        The mean of the distribution is the estimated state and the covariance
        is the covariance of the estimate.

    """
    if state_count is None:
        state_count = kalman_filter.state_count

    state_count = int(state_count)
    if state_count < 0:
        raise ValueError("Invalid final time step: {}".format(state_count))

    # No states to return?
    if state_count == 0:
        return []

    # Initialise with final posterior estimate
    states = [None] * state_count
    states[-1] = kalman_filter.posterior_state_estimates[-1]

    priors = kalman_filter.prior_state_estimates
    posteriors = kalman_filter.posterior_state_estimates

    # Work backwards from final state
    for k in range(state_count-2, -1, -1):
        process_mat = kalman_filter.process_matrices[k+1]
        cmat = posteriors[k].cov.dot(process_mat.T).dot(
            np.linalg.inv(priors[k+1].cov))

        # Calculate smoothed state and covariance
        states[k] = MultivariateNormal(
            mean=posteriors[k].mean + cmat.dot(states[k+1].mean -
                                               priors[k+1].mean),
            cov=posteriors[k].cov + cmat.dot(states[k+1].cov -
                                             priors[k+1].cov).dot(cmat.T)
        )

    return states
