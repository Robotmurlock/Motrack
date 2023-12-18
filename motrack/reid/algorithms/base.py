"""
Interface for re-identification appearance extractor
"""
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from motrack.library.cv.bbox import BBox


class BaseReID(ABC):
    """.
    Interface for re-identification appearance extractor
    """
    @abstractmethod
    def extract_features(self, frame: np.ndarray, frame_index: int, scene: Optional[str] = None) -> np.ndarray:
        """
        Extracts object appearance features for given frame.

        Args:
            frame: Raw frame
            frame_index: Frame number (index)
            scene: Scene name (optional, for caching)

        Returns:
            Object frame appearance features
        """

    def extract_objects_features(self, frame: np.ndarray, bboxes: List[BBox], frame_index: int, scene: Optional[str] = None) -> np.ndarray:
        """
        Performs object appearance features for given frame and list of objects' bounding boxes.

        Args:
            frame: Raw frame
            bboxes: Detected bounding boxes
            frame_index: Frame number (index)
            scene: Scene name (optional, for caching)

        Returns:
            Array of object appearance features for each detected object
        """
        objects_features: List[np.ndarray] = []

        for bbox in bboxes:
            crop = bbox.crop(frame)
            object_features = self.extract_features(crop, frame_index=frame_index, scene=scene)
            objects_features.append(object_features)

        return np.concatenate(objects_features, 0)
