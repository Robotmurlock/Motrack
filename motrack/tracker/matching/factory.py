"""
Tracker association factory method.
Use `ASSOCIATION_CATALOG.register` to extend supported tracker association algorithms.
"""
from typing import Dict, Any

from motrack.tracker.matching.algorithms.base import AssociationAlgorithm
# noinspection PyUnresolvedReferences
from motrack.tracker.matching.algorithms.biou import HungarianCBIoU, HungarianBIoU
# noinspection PyUnresolvedReferences
from motrack.tracker.matching.algorithms.dcm import DCMIoU, MoveDCM
# noinspection PyUnresolvedReferences
from motrack.tracker.matching.algorithms.iou import IoUAssociation
# noinspection PyUnresolvedReferences
from motrack.tracker.matching.algorithms.move import Move
# noinspection PyUnresolvedReferences
from motrack.tracker.matching.algorithms.reid_iou import ReidIoUAssociation
from motrack.tracker.matching.catalog import ASSOCIATION_CATALOG


def association_factory(name: str, params: Dict[str, Any]) -> AssociationAlgorithm:
    """
    Association algorithm factory. Not case-sensitive.

    Args:
        name: Algorithm name
        params: Parameters

    Returns:
        AssociationAlgorithm object
    """
    return ASSOCIATION_CATALOG[name](**params)
