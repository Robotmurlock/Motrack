"""
Interface for re-identification appearance extractor
"""
from abc import ABC, abstractmethod
from typing import List

import numpy as np

from motrack.library.cv.bbox import BBox


class BaseReID(ABC):
    """.
    Interface for re-identification appearance extractor
    """
    @abstractmethod
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extracts object appearance features for given image.

        Returns:
            Object image appearance features
        """

    def extract_objects_features(self, image: np.ndarray, bboxes: List[BBox]) -> np.ndarray:
        """
        Performs object appearance features for given image and list of objects' bounding boxes.

        Args:
            image: Raw image
            bboxes: Detected bounding boxes

        Returns:
            Array of object appearance features for each detected object
        """
        objects_features: List[np.ndarray] = []

        for bbox in bboxes:
            crop = bbox.crop(image)
            object_features = self.extract_features(crop)
            objects_features.append(object_features)

        return np.concatenate(objects_features, 0)
