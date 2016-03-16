"""
General utility functions used internally.

"""
import numpy as np

def as_square_array(arr):
    """Return arr massaged into a square array. Raises ValueError if arr cannot be
    so massaged.

    """
    arr = np.atleast_2d(arr)
    if len(arr.shape) != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Expected square array")
    return arr
