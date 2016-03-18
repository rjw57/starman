"""
Helpers for linear systems.

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
