"""
YOLOv8 inference support.
"""
from typing import List, Optional, Tuple, Union, Any

import cv2
import numpy as np

from motrack.library.cv.bbox import LabelType
from motrack.object_detection.algorithms.base import ObjectDetectionInference
from motrack.object_detection.catalog import OBJECT_DETECTION_CATALOG
from motrack.utils.lookup import LookupTable


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
        is_rgb: bool = True
    ):
        super().__init__(lookup=lookup)
        try:
            import ultralytics
        except ImportError as e:
            raise ImportError('Please install ultralytics package with "pip3 install ultralytics" in order to use YOLOv8!') from e

        self._yolo = ultralytics.YOLO(model_path)
        self._yolo.to(accelerator)
        self._verbose = verbose
        self._conf = conf
        self._class_filter = class_filter
        self._is_rgb = is_rgb

    def predict_raw(self, image: np.ndarray) -> Any:
        if self._is_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return self._yolo.predict(
            source=image,
            verbose=self._verbose,
            conf=self._conf,
            classes=self._class_filter
        )

    def postprocess(self, image: np.ndarray, raw: Any) -> Tuple[Union[np.ndarray, List[float]], List[LabelType], Union[np.ndarray, list]]:
        raw = raw[0]  # Remove batch
        h, w, _ = image.shape

        # Process bboxes
        bboxes = raw.boxes.xyxy.detach().cpu().numpy()
        bboxes[:, [0, 2]] /= w
        bboxes[:, [1, 3]] /= h

        # Process classes
        classes = raw.boxes.cls.detach().cpu().numpy()

        # Process confidences
        confidences = raw.boxes.conf.detach().cpu().view(-1).numpy()

        return bboxes, classes, confidences
