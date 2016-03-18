import pytest
import numpy as np
from starman.linearsystem import measure_states, generate_states

def test_measure_states():
    H = np.array([1, 0])
    R = np.eye(1)
    states = np.random.multivariate_normal(
        mean=np.zeros(2), cov=np.eye(2), size=4
    )
    assert states.shape == (4, 2)

    assert measure_states(np.zeros((0, 2)), H, R).shape == (0, 1)
    assert measure_states(states, H, R).shape == (4, 1)

def test_measure_states_validates_args():
    H = np.array([1, 0, 0])
    R = np.eye(2)
    with pytest.raises(ValueError):
        measure_states(np.zeros((0, 3)), H, R)

def test_generate_states():
    F = np.array([[1, 1], [0, 1]])
    Q = np.eye(2) * 1e-2
    states = generate_states(10, F, Q)
    assert states.shape == (10, 2)
