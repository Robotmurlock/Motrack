from typing import List, Optional

import numpy as np
import ultralytics

from motrack.library.cv.bbox import PredBBox, LabelType
from motrack.object_detection.algorithms.base import ObjectDetectionInference
from motrack.utils.lookup import LookupTable
from motrack.object_detection.catalog import OBJECT_DETECTION_CATALOG


@OBJECT_DETECTION_CATALOG.register('yolov8')
class YOLOv8Inference(ObjectDetectionInference):
    """
    Object Detection - YOLOv8
    """
    def __init__(
        self,
        model_path: str,
        accelerator: str,
        verbose: bool = False,
        conf: float = 0.25,
        class_filter: Optional[List[LabelType]] = None,
        lookup: Optional[LookupTable] = None,
    ):
        super().__init__(lookup=lookup)
        self._yolo = ultralytics.YOLO(model_path)
        self._yolo.to(accelerator)
        self._verbose = verbose
        self._conf = conf
        self._class_filter = class_filter
        if self._class_filter is None:
            self._class_filter = []

    def predict(
        self,
        image: np.ndarray
    ) -> List[PredBBox]:
        h, w, _ = image.shape
        prediction_raw = self._yolo.predict(
            source=image,
            verbose=self._verbose,
            conf=self._conf,
            classes=self._class_filter
        )[0]  # Remove batch

        # Process bboxes
        bboxes = prediction_raw.boxes.xyxy.detach().cpu()
        bboxes[:, [0, 2]] /= w
        bboxes[:, [1, 3]] /= h

        # Process classes
        class_indices = prediction_raw.boxes.cls.detach().cpu()
        classes = [self._lookup.inverse_lookup(int(cls_index)) for cls_index in class_indices.view(-1)]

        # Process confidences
        confidences = prediction_raw.boxes.conf.detach().cpu().view(-1)

        return self.pack_bboxes(bboxes, classes, confidences)
