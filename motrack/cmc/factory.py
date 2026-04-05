"""
CMC factory method.
Use `CMC_CATALOG.register` to extend supported CMC algorithms.
"""
# noinspection PyUnresolvedReferences
from motrack.cmc.algorithms.gmc_from_file import GMCFromFile  # pylint: disable=unused-import
from motrack.cmc.algorithms.base import CameraMotionCompensation
from motrack.cmc.catalog import CMC_CATALOG

CMC_CATALOG.validate()


def cmc_factory(name: str, params: dict) -> CameraMotionCompensation:
    """
    CMC factory.

    Args:
        name: Algorithm name
        params: Dataset creation parameters

    Returns:
        Created CMC object

    Raises:
        TypeError: If CMC params are not a dictionary or None.
        ValueError: If CMC params are invalid.
    """
    return CMC_CATALOG.create(name, params, params_label='cmc')
