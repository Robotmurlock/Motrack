"""
Tracker association factory method.
Use `ASSOCIATION_CATALOG.register` to extend supported tracker association algorithms.
"""
from typing import Dict, Any

from motrack.tracker.matching.algorithms.base import AssociationAlgorithm
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
