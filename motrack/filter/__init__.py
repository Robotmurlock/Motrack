"""
Filter interface.
"""
from motrack.filter.algorithms.base import StateModelFilter
from motrack.filter.algorithms.kalman_filter import BotSortKalmanWrapFilter
from motrack.filter.algorithms.no_motion import NoMotionFilter
from motrack.filter.factory import filter_factory
