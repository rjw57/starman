"""
Rauch-Tung-Striebel smoother for Kalman filters.

"""
# pylint:disable=redefined-builtin
from builtins import range

import numpy as np

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
        (tuple): 2-element tuple containing:

        - **states** (*array*): An NxSTATE_DIM array of smoothed state vectors.
        - **state_covariances** (*array*): An NxSTATE_DIMxSTATE_DIM array of
          smoothed state covariances.

    """
    state_dim = kalman_filter.state_length
    if state_count is None:
        state_count = kalman_filter.state_count

    state_count = int(state_count)
    if state_count < 0:
        raise ValueError("Invalid final time step: {}".format(state_count))

    states = np.nan * np.ones((state_count, state_dim))
    state_covariances = np.nan * np.ones((state_count, state_dim, state_dim))

    # Initialise with final posterior estimate
    states[-1, ...] = kalman_filter.posterior_state_estimates[-1]
    state_covariances[-1, ...] = kalman_filter.posterior_state_covariances[-1]

    # Work backwards from final state
    for k in range(state_count-2, -1, -1):
        process_mat = kalman_filter.process_matrices[k+1]
        cmat = kalman_filter.posterior_state_covariances[k].dot(process_mat.T).dot(
            np.linalg.inv(kalman_filter.prior_state_covariances[k+1])
        )

        # Calculate smoothed state and covariance
        states[k, ...] = kalman_filter.posterior_state_estimates[k] + \
            cmat.dot(states[k+1, ...] - kalman_filter.prior_state_estimates[k+1])
        state_covariances[k, ...] = kalman_filter.posterior_state_covariances[k] + \
            cmat.dot(
                state_covariances[k+1, ...] - kalman_filter.prior_state_covariances[k+1]
            ).dot(cmat.T)

    return states, state_covariances
