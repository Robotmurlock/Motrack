"""
CMC factory method.
Use `CMC_CATALOG.register` to extend supported CMC algorithms.
"""
from motrack.cmc.algorithms.base import CameraMotionCompensation
# noinspection PyUnresolvedReferences
from motrack.cmc.algorithms.gmc_from_file import GMCFromFile  # pylint: disable=unused-import
from motrack.cmc.catalog import CMC_CATALOG


def cmc_factory(name: str, params: dict) -> CameraMotionCompensation:
    """
    CMC factory.

    Args:
        name: Algorithm name
        params: Dataset creation parameters

    Returns:
        Created CMC object
    """
    return CMC_CATALOG[name](**params)
