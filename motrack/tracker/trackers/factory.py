"""
Tracker factory method.
Use `TRACKER_CATALOG.register` to extend supported tracker algorithms.
"""
from motrack.tracker.trackers.algorithms.base import Tracker
# noinspection PyUnresolvedReferences
from motrack.tracker.trackers.algorithms.byte import ByteTracker  # pylint: disable=unused-import
# noinspection PyUnresolvedReferences
from motrack.tracker.trackers.algorithms.sort import SortTracker  # pylint: disable=unused-import
# noinspection PyUnresolvedReferences
from motrack.tracker.trackers.algorithms.sparse import SparseTracker  # pylint: disable=unused-import
# noinspection PyUnresolvedReferences
from motrack.tracker.trackers.algorithms.fast import FastTracker  # pylint: disable=unused-import
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
