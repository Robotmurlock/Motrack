"""
Filter factory method.
Use `FILTER_CATALOG.register` to extend supported filter algorithms.
"""
# noinspection PyUnresolvedReferences
from motrack.filter.algorithms.end_to_end import MotrackMotionFilterEndToEnd  # pylint: disable=unused-import
# noinspection PyUnresolvedReferences
from motrack.filter.algorithms.kalman_filter import BotSortKalmanWrapFilter  # pylint: disable=unused-import
# noinspection PyUnresolvedReferences
from motrack.filter.algorithms.no_motion import NoMotionFilter  # pylint: disable=unused-import
from motrack.filter.algorithms.base import StateModelFilter
from motrack.filter.catalog import FILTER_CATALOG

FILTER_CATALOG.validate()


def filter_factory(name: str, params: dict) -> StateModelFilter:
    """
    Filter (StateModel) factory.

    Args:
        name: Filter name (type)
        params: Filter creation parameters

    Returns:
        Created filter object

    Raises:
        TypeError: If filter params are not a dictionary or None.
        ValueError: If filter params are invalid.
    """
    return FILTER_CATALOG.create(name, params, params_label='filter')
