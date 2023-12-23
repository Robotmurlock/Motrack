"""
Inference support for FastReId framework.
Reference: https://github.com/JDAI-CV/fast-reid
"""
from typing import Optional

import cv2
import numpy as np

from motrack.reid.algorithms.base import BaseReID
from motrack.reid.catalog import REID_CATALOG


@REID_CATALOG.register('fastreid-onnx')
class FastReIDOnnx(BaseReID):
    """
    FastReIDOnnx framework support for any models exported in ONNX format.
    """
    def __init__(self, model_path: str, height: int = 256, width: int = 256, cache_path: Optional[str] = None):
        """
        Args:
            model_path: Path where the ONNX model export is stored
            height: Image input height
            width: Image input width
            cache_path: Inference cache path
        """
        super().__init__(cache_path=cache_path)

        import onnxruntime
        self._ort_session = onnxruntime.InferenceSession(model_path)
        self._input_name = self._ort_session.get_inputs()[0].name
        self._height = height
        self._width = width

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocesses the input frame. Ref: https://github.com/JDAI-CV/fast-reid/blob/master/tools/deploy/onnx_inference.py

        Args:
            frame: Object crop

        Returns:
            frame features (not normalized)
        """
        # the model expects RGB inputs
        frame = frame[:, :, ::-1]

        # Apply pre-processing to frame.
        frame = cv2.resize(frame, (self._width, self._height), interpolation=cv2.INTER_CUBIC)
        frame = frame.astype(np.float32).transpose(2, 0, 1)[np.newaxis]  # (1, 3, h, w)
        return frame

    # noinspection PyMethodMayBeStatic
    def _postprocess(self, features: np.ndarray) -> np.ndarray:
        """
        Normalized the input features to L2 ball.

        Args:
            features: Un-normalized features

        Returns:
            Normalized features
        """
        norm = np.linalg.norm(features, ord=2, axis=1, keepdims=True)
        return features / (norm + np.finfo(np.float32).eps)

    def extract_features(self, frame: np.ndarray, frame_index: int, scene: Optional[str] = None) -> np.ndarray:
        frame = self._preprocess(frame)
        features = self._ort_session.run(None, {self._input_name: frame})[0]
        return self._postprocess(features)
