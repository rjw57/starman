import numpy as np
import pytest
from starman import MultivariateNormal

def test_mv_no_args():
    mv = MultivariateNormal()
    assert mv.mean.shape == (1,)
    assert mv.mean[0] == 0.0
    assert mv.cov.shape == (1, 1)
    assert mv.cov[0, 0] == 1.0

def test_mv_mean_only():
    mv1 = MultivariateNormal(mean=1)
    assert mv1.mean.shape == (1,)
    assert mv1.cov.shape == (1, 1)

    mv2 = MultivariateNormal(mean=[1,2])
    assert mv2.mean.shape == (2,)
    assert mv2.cov.shape == (2, 2)

def test_mv_cov_only():
    mv1 = MultivariateNormal(cov=np.eye(2))
    assert mv1.mean.shape == (2,)

    mv2 = MultivariateNormal(cov=1)
    assert mv2.cov.shape == (1, 1)

def test_mv_invalid_args():
    with pytest.raises(ValueError):
        # Mean should be a vector
        MultivariateNormal(mean=np.eye(3))

    with pytest.raises(ValueError):
        # Covariance should be square
        MultivariateNormal(cov=[[1, 2]])

def test_random_sample():
    np.random.seed(0xdeadbeef)
    mv = MultivariateNormal(mean=[1,2])
    samples = mv.rvs(300)
    assert samples.shape == (300, 2)

    delta = np.abs(np.mean(samples, axis=0) - mv.mean)
    assert np.all(delta < 1e-1)
