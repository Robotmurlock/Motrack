"""
Oracle inference support.
"""
from typing import List, Tuple, Union, Any

import numpy as np

from motrack.library.cv.bbox import LabelType
from motrack.object_detection.algorithms.base import ObjectDetectionInference
from motrack.object_detection.catalog import OBJECT_DETECTION_CATALOG


@OBJECT_DETECTION_CATALOG.register('oracle')
class OracleInference(ObjectDetectionInference):
    """
    Oracle inference - Dummy inference object in case oracle detections are used
    """

    def predict_raw(self, image: np.ndarray) -> Any:
        raise NotImplementedError('Oracle "inference" can\'t perform inference. Did you forgot to set oracle=True in detection manager?')

    def postprocess(self, image: np.ndarray, raw: Any) -> Tuple[Union[np.ndarray, List[float]], List[LabelType], Union[np.ndarray, list]]:
        raise NotImplementedError('Oracle "inference" can\'t perform inference. Did you forgot to set oracle=True in detection manager?')