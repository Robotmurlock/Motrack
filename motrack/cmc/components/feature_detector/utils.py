"""
Shared helpers for feature detectors.
"""
from typing import List, Optional

import numpy as np


def empty_points() -> np.ndarray:
    """
    Creates an empty point array, returned when a detector finds nothing.

    Returns:
        Empty points of shape (0, 2)
    """
    return np.zeros((0, 2), dtype=np.float32)


def pack_keypoints(keypoints: List, descriptors: Optional[np.ndarray]) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Converts OpenCV keypoints into a point array, keeping them aligned with the descriptors.

    Args:
        keypoints: OpenCV keypoints
        descriptors: Descriptors matching the keypoints, or None

    Returns:
        Points of shape (N, 2) and their descriptors
    """
    if not keypoints or descriptors is None:
        return empty_points(), None

    points = np.array([keypoint.pt for keypoint in keypoints], dtype=np.float32)
    return points, descriptors
