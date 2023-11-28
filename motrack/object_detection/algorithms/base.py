from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np

from motrack.library.cv.bbox import PredBBox, BBox, LabelType
from motrack.utils.lookup import LookupTable


class ObjectDetectionInference(ABC):
    """
    ObjectDetection inference interface.
    """
    def __init__(self, lookup: Optional[LookupTable] = None):
        self._lookup = lookup

    @abstractmethod
    def predict(
        self,
        image: np.ndarray
    ) -> List[PredBBox]:
        """
        Generates list of bbox prediction based on given image data

        Args:
            image: Image data

        Returns:
            List of bbox predictions
        """
        pass

    @staticmethod
    def pack_bboxes(
        bbox_xyxy: Union[np.ndarray, List[float]],
        classes: List[LabelType],
        confidences: Union[np.ndarray, list]
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
        pred_bboxes: List[PredBBox] = []
        for bbox_data, cls, conf in zip(bbox_xyxy, classes, confidences):
            pred_bbox = PredBBox.create(
                bbox=BBox.from_xyxy(*bbox_data),
                label=cls,
                conf=conf
            )
            pred_bboxes.append(pred_bbox)

        return pred_bboxes
