"""
starman is a module which provides implementations of popular state estimation
and tracking algorithms.

"""

from .stateestimation import KalmanFilter, GaussianStateEstimation
from .rts import rts_smooth
from .stats import MultivariateNormal
from .slh import slh_associate
