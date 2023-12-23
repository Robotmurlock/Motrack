"""
Inference support for FastReId framework.
Reference: https://github.com/JDAI-CV/fast-reid
"""
import logging
from typing import Optional

import cv2
import numpy as np

from motrack.reid.algorithms.base import BaseReID
from motrack.reid.catalog import REID_CATALOG

logger = logging.getLogger('FastReIDOnnx')


@REID_CATALOG.register('fastreid-onnx')
class FastReIDOnnx(BaseReID):
    """
    FastReIDOnnx framework support for any models exported in ONNX format.
    """
    def __init__(
        self,
        model_path: str,
        height: int = 256,
        width: int = 256,
        cache_path: Optional[str] = None,
        batch_inference: bool = False
    ):
        """
        Args:
            model_path: Path where the ONNX model export is stored
            height: Image input height
            width: Image input width
            cache_path: Inference cache path
            batch_inference: Use batch inference mode if supported by the ONNX export
                - This makes inference faster
        """
        super().__init__(cache_path=cache_path, batch_inference=batch_inference)

        import onnxruntime
        logger.info(f'Runtime device: {onnxruntime.get_device()}.')
        self._ort_session = onnxruntime.InferenceSession(model_path)
        self._input_name = self._ort_session.get_inputs()[0].name
        self._height = height
        self._width = width

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        # the model expects RGB inputs
        image = image[:, :, ::-1]

        # Apply pre-processing to frame.
        image = cv2.resize(image, (self._width, self._height), interpolation=cv2.INTER_CUBIC)
        image = image.astype(np.float32).transpose(2, 0, 1)  # (1, 3, h, w)
        return image

    def postprocess(self, features: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(features, ord=2, axis=1, keepdims=True)
        return features / (norm + np.finfo(np.float32).eps)

    def inference(self, image: np.ndarray) -> np.ndarray:
        return self._ort_session.run(None, {self._input_name: image})[0]
