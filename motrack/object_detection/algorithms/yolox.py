from typing import List, Optional

import numpy as np

from motrack.library.cv.bbox import PredBBox
from motrack.object_detection.algorithms.base import ObjectDetectionInference
from motrack.object_detection.catalog import OBJECT_DETECTION_CATALOG
from motrack.object_detection.yolox import YOLOXPredictor
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

        self._yolox = YOLOXPredictor(
            checkpoint_path=model_path,
            accelerator=accelerator,
            conf_threshold=conf,
            legacy=legacy
        )
        self._conf = conf
        self._min_bbox_area = min_bbox_area

    def predict(self, image: np.ndarray) -> List[PredBBox]:
        h, w, _ = image.shape

        output, _ = self._yolox.predict(image)

        # Filter small bboxes
        bboxes = output[:, :4]
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        output = output[areas >= self._min_bbox_area]

        # Process bboxes
        bboxes = output[:, :4]
        bboxes[:, [0, 2]] /= w
        bboxes[:, [1, 3]] /= h

        # Process classes
        class_indices = output[:, 6]
        classes = [self._lookup.inverse_lookup(int(cls_index)) for cls_index in class_indices] \
            if self._lookup is not None else class_indices

        # Process confidences
        confidences = output[:, 4]


        return self.pack_bboxes(bboxes, classes, confidences)
