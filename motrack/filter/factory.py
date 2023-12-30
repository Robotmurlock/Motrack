"""
Filter factory method.
Use `FILTER_CATALOG.register` to extend supported filter algorithms.
"""
from motrack.filter.algorithms.base import StateModelFilter
# noinspection PyUnresolvedReferences
from motrack.filter.algorithms.end_to_end import MotrackMotionFilterEndToEnd  # pylint: disable=unused-import
# noinspection PyUnresolvedReferences
from motrack.filter.algorithms.kalman_filter import BotSortKalmanWrapFilter  # pylint: disable=unused-import
# noinspection PyUnresolvedReferences
from motrack.filter.algorithms.no_motion import NoMotionFilter  # pylint: disable=unused-import
from motrack.filter.catalog import FILTER_CATALOG


def filter_factory(name: str, params: dict) -> StateModelFilter:
    """
    Filter (StateModel) factory.

    Args:
        name: Filter name (type)
        params: Filter creation parameters

    Returns:
        Created filter object
    """
    return FILTER_CATALOG[name](**params)
