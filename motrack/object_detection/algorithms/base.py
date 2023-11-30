"""
Object detection inference interface.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Any

import numpy as np

from motrack.library.cv.bbox import PredBBox, BBox
from motrack.utils.lookup import LookupTable


class ObjectDetectionInference(ABC):
    """
    ObjectDetection inference interface.
    """
    def __init__(self, lookup: Optional[LookupTable] = None):
        self._lookup = lookup

    @abstractmethod
    def predict_raw(self, image: np.ndarray) -> Any:
        """
        Performs inference and returns raw model output.

        Args:
            image: Image data

        Returns:
            Raw model output
        """

    @abstractmethod
    def postprocess(self, image: np.ndarray, raw: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Performs postprocess of the raw model output.

        Args:
            image: Image data
            raw: Raw model output

        Returns:
            - List of object bboxes arrays in format xyxy
            - List of object classes
            - List of object bbox confidences
        """

    def predict_with_postprocess(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates list of raw bbox prediction based on given image data.

        Args:
            image: Image data

        Returns:
            - List of object bboxes arrays in format xyxy
            - List of object classes (ints)
            - List of object bbox confidences
        """
        raw = self.predict_raw(image)
        return self.postprocess(image, raw)

    def predict_bboxes(self, image: np.ndarray) -> List[PredBBox]:
        """
        Generates list of bbox prediction based on given image data

        Args:
            image: Image data

        Returns:
            List of bbox predictions
        """
        bboxes, classes, confidences = self.predict_with_postprocess(image)
        return self.pack_bboxes(bboxes, classes, confidences)

    def pack_bboxes(
        self,
        bbox_xyxy: np.ndarray,
        classes: np.ndarray,
        confidences: np.ndarray
    ) -> List[PredBBox]:
        """
        Packs OD raw data into bboxes

        Args:
            bbox_xyxy: BBox in xyxy coordinate format
            classes: Classes (int or str)
            confidences: Confidences

        Returns:
            Returns list of bboxes
        """
        classes = [int(cls) for cls in classes]
        confidences = [float(conf) for conf in confidences]
        if self._lookup is not None:
            classes = [self._lookup.inverse_lookup(cls) for cls in classes]

        pred_bboxes: List[PredBBox] = []
        for bbox_data, cls, conf in zip(bbox_xyxy, classes, confidences):
            pred_bbox = PredBBox.create(
                bbox=BBox.from_xyxy(*bbox_data),
                label=cls,
                conf=conf
            )
            pred_bboxes.append(pred_bbox)

        return pred_bboxes
