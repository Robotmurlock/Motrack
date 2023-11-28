"""
Tracker factory method.
"""
from motrack.tracker.trackers.algorithms.base import Tracker
# noinspection PyUnresolvedReferences
from motrack.tracker.trackers.algorithms.byte import ByteTracker
# noinspection PyUnresolvedReferences
from motrack.tracker.trackers.algorithms.sort import SortTracker
# noinspection PyUnresolvedReferences
from motrack.tracker.trackers.algorithms.sparse import SparseTracker
from motrack.tracker.trackers.catalog import TRACKER_CATALOG


def tracker_factory(name: str, params: dict) -> Tracker:
    """
    Tracker factory

    Args:
        name: tracker name
        params: tracker params

    Returns:
        Tracker object
    """
    return TRACKER_CATALOG[name](**params)