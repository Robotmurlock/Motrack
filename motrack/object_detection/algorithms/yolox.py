"""
YOLOX inference class.
"""
from typing import List, Optional, Tuple, Union, Any

import numpy as np

from motrack.library.cv.bbox import LabelType
from motrack.object_detection.algorithms.base import ObjectDetectionInference
from motrack.object_detection.catalog import OBJECT_DETECTION_CATALOG
from motrack.utils.lookup import LookupTable


@OBJECT_DETECTION_CATALOG.register('yolox')
class YOLOXInference(ObjectDetectionInference):
    """
    Object Detection - YOLOX
    """
    def __init__(
        self,
        model_path: str,
        accelerator: str,
        conf: float = 0.01,
        min_bbox_area: int = 0,
        legacy: bool = True,
        lookup: Optional[LookupTable] = None
    ):
        super().__init__(lookup=lookup)
        try:
            from motrack.object_detection.yolox import YOLOXPredictor
        except ImportError as e:
            raise ImportError('Please install PyTorch from "https://pytorch.org/"'
                              'and YOLOX package from "https://github.com/Megvii-BaseDetection/YOLOX"!') from e

        self._yolox = YOLOXPredictor(
            checkpoint_path=model_path,
            accelerator=accelerator,
            conf_threshold=conf,
            legacy=legacy
        )
        self._min_bbox_area = min_bbox_area

    def predict_raw(self, image: np.ndarray) -> Any:
        output, _ = self._yolox.predict(image)
        return output

    def postprocess(self, image: np.ndarray, raw: Any) -> Tuple[Union[np.ndarray, List[float]], List[LabelType], Union[np.ndarray, list]]:
        h, w, _ = image.shape

        # Filter small bboxes
        bboxes = raw[:, :4]
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        output = raw[areas >= self._min_bbox_area]

        # Process bboxes
        bboxes = output[:, :4]
        bboxes[:, [0, 2]] /= w
        bboxes[:, [1, 3]] /= h

        # Process classes
        classes = output[:, 6]

        # Process confidences
        confidences = output[:, 4]

        return bboxes, classes, confidences
