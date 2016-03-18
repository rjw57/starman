"""
The :py:mod:`starman.linearsystem` module contains some helper functions for
systems with linear dynamics and a linear measurement model.

"""
import numpy as np

def measure_states(states, measurement_matrix, measurement_covariance):
    """
    Measure a list of states with a measurement matrix in the presence of
    measurement noise.

    Args:
        states (array): states to measure. Shape is NxSTATE_DIM.
        measurement_matrix (array): Each state in *states* is measured with this
            matrix. Should be MEAS_DIMxSTATE_DIM in shape.
        measurement_covariance (array): Measurement noise covariance. Should be
            MEAS_DIMxMEAS_DIM.

    Returns:
        (array): NxMEAS_DIM array of measurements.

    """
    # Sanitise input
    measurement_matrix = np.atleast_2d(measurement_matrix)
    measurement_covariance = np.atleast_2d(measurement_covariance)
    measurement_dim = measurement_matrix.shape[0]
    if measurement_covariance.shape != (measurement_dim, measurement_dim):
        raise ValueError(("Measurement matrix and covariance have inconsistent "
                          "shapes {} and {}").format(measurement_matrix.shape,
                                                     measurement_covariance.shape))
    states = np.atleast_2d(states)

    # Special case: no output
    if states.shape[0] == 0:
        return np.zeros((0, measurement_dim))

    # Measure states
    measurement_means = measurement_matrix.dot(states.T).T
    measurement_noises = np.random.multivariate_normal(
        mean=np.zeros(measurement_dim), cov=measurement_covariance,
        size=states.shape[0]
    )

    return measurement_means + measurement_noises

def generate_states(state_count, process_matrix, process_covariance,
                    initial_state=None):
    """
    Generate states by simulating a linear system with constant process matrix
    and process noise covariance.

    Args:
        state_count (int): Number of states to generate.
        process_matrix (array): Square array
        process_covariance (array): Square array specifying process noise
            covariance.
        initial_state (array or None): If omitted, use zero-filled vector as
            initial state.

    """
    # Sanitise input
    process_matrix = np.atleast_2d(process_matrix)
    process_covariance = np.atleast_2d(process_covariance)
    state_dim = process_matrix.shape[0]

    if process_matrix.shape != (state_dim, state_dim):
        raise ValueError("Process matrix has inconsistent shape: {}".format(
            process_matrix.shape))

    if process_covariance.shape != (state_dim, state_dim):
        raise ValueError("Process covariance has inconsistent shape: {}".format(
            process_covariance.shape))

    if initial_state is None:
        initial_state = np.zeros(process_matrix.shape[0])

    states = [initial_state]
    while len(states) < state_count:
        states.append(
            process_matrix.dot(states[-1]) + np.random.multivariate_normal(
                mean=np.zeros(state_dim), cov=process_covariance
            )
        )

    return np.vstack(states)
