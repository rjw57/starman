"""
Exceptions used by starman.

"""

class ParameterError(ValueError):
    """Represents in invalid or erroneous parameter passed to a function or
    method.

    """

class NoAPrioriStateError(RuntimeError):
    """An attempt was made to update an *a posteriori* state estimate with no
    corresponding *a priori* state estimate available.

    """
