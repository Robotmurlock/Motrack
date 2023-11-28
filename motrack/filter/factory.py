# noinspection PyUnresolvedReferences
from motrack.filter.algorithms.kalman_filter import BotSortKalmanWrapFilter
# noinspection PyUnresolvedReferences
from motrack.filter.algorithms.no_motion import NoMotionFilter
from motrack.filter.algorithms.base import StateModelFilter
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
