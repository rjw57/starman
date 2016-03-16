import numpy as np
import pytest

import starman.util as util

def test_as_square_array():
    assert util.as_square_array(1).shape == (1, 1)
    assert util.as_square_array([1]).shape == (1, 1)
    assert util.as_square_array([[1, 2], [3, 4]]).shape == (2, 2)

def test_as_square_array_needs_square_input():
    with pytest.raises(ValueError):
        util.as_square_array([
            [ [1,2,3], [4,5,6] ],
            [ [7,8,9], [0,1,2] ],
        ])

    with pytest.raises(ValueError):
        util.as_square_array([1, 2])

    with pytest.raises(ValueError):
        util.as_square_array([[1, 2], [3, 4], [5, 6]])

