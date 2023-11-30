"""
Object detection factory method.
Use `OBJECT_DETECTION_CATALOG.register` to extend supported object detection algorithms.
"""
from typing import Optional

from motrack.object_detection.algorithms.base import ObjectDetectionInference
from motrack.object_detection.catalog import OBJECT_DETECTION_CATALOG
from motrack.utils.lookup import LookupTable


def object_detection_inference_factory(
    name: str,
    params: dict,
    lookup: Optional[LookupTable] = None
) -> ObjectDetectionInference:
    """
    Creates object detection inference for single object tracking (filter evaluation) given name and parameters.

    Args:
        name: OD inference name (type)
        params: CTor parameters
        lookup: Used to decode classes

    Returns:
        Initialized OD inference object for filter evaluation
    """
    return OBJECT_DETECTION_CATALOG[name](**params, lookup=lookup)
