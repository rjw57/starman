"""
Statistics.

"""
import numpy as np

class MultivariateNormal(object):
    """
    MultivariateNormal represents a multivariate normal (or "Gaussian")
    distribution parametrised in terms of a mean and covariance. The mean is a
    length-N vector and the covariance is a NxN matrix.

    If mean is unspecified, it defaults to a zero-filled vector whose dimension
    matches the covariance.

    If the covariance is unspecified, it defaults to an identity matrix whose
    shape matches the dimension of the mean.

    If neither mean or covariance are specified, default values of 0 and 1 are
    used.

    Args:
        mean (None or array): Distribution mean.
        cov (None or array): Distribution covariance.

    """
    def __init__(self, mean=None, cov=None):
        if mean is None and cov is None:
            mean, cov = np.array([1.0]), np.array([[1.0]])
        elif mean is not None and cov is None:
            mean = np.atleast_1d(mean)
            cov = np.eye(mean.shape[0])
        elif cov is not None and mean is None:
            cov = np.atleast_2d(cov)
            mean = np.zeros(cov.shape[0])
        else:
            mean = np.atleast_1d(mean)
            cov = np.atleast_2d(cov)

        if cov.shape[0] != cov.shape[1]:
            raise ValueError('Covariance must be square')

        if len(mean.shape) != 1:
            raise ValueError('Mean must be 1 dimensional')

        self.mean, self.cov = mean, cov

    def rvs(self, size=1):
        """Convenience method to sample from this distribution.

        Args:
            size (int or tuple): Shape of return value. Each element is drawn
                independently from this distribution.

        """
        return np.random.multivariate_normal(self.mean, self.cov, size)

