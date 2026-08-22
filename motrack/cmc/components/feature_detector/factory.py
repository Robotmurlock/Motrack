"""
Feature detector factory method.
Use `FEATURE_DETECTOR_CATALOG.register` to extend supported feature detectors.
"""
from motrack.cmc.components.feature_detector.algorithms.base import FeatureDetector
from motrack.cmc.components.feature_detector.catalog import FEATURE_DETECTOR_CATALOG


def feature_detector_factory(detector_type: str, params: dict) -> FeatureDetector:
    """
    Feature detector factory.

    Args:
        detector_type: Feature detector type
        params: Feature detector params

    Returns:
        Feature detector object

    Raises:
        TypeError: If feature detector params are not a dictionary or None.
        RuntimeError: If feature detector config models and registered detectors are out of sync.
        ValueError: If the feature detector type or params are invalid.
    """
    return FEATURE_DETECTOR_CATALOG.create(detector_type, params, params_label='feature detector')
