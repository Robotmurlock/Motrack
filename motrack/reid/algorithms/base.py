"""
Interface for re-identification appearance extractor
"""
import os
from abc import ABC, abstractmethod
from typing import List, Optional
import copy

import numpy as np

from motrack.library.cv.bbox import BBox
from motrack.library.numpy_utils.io import load_npy, store_npy


class BaseReID(ABC):
    """.
    Interface for re-identification appearance extractor
    """
    def __init__(self, cache_path: Optional[str] = None, batch_inference: bool = False):
        """
        Args:
            cache_path: Inference cache path
            batch_inference: Use batch inference mode if supported by the ONNX export
                - This makes inference faster
        """
        self._cache_path = cache_path
        self._batch_inference = batch_inference

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocesses the input frame. Ref: https://github.com/JDAI-CV/fast-reid/blob/master/tools/deploy/onnx_inference.py

        Args:
            image: Object crop

        Returns:
            frame features (not normalized)
        """

    @abstractmethod
    def postprocess(self, features: np.ndarray) -> np.ndarray:
        """
        Normalized the input features to L2 ball.

        Args:
            features: Un-normalized features

        Returns:
            Normalized features
        """

    @abstractmethod
    def inference(self, image: np.ndarray) -> np.ndarray:
        """
        Extracts object appearance features for given frame.

        Args:
            image: Raw frame

        Returns:
            Object frame appearance features
        """

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        data = self.preprocess(image)
        features = self.inference(data)
        return self.postprocess(features)

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
        if self._cache_path is not None:
            # Load features from cache instead of running the inference
            assert scene is not None, 'Scene must be specified in order to perform caching!'
            scene_frame_cache_path = self._get_scene_frame_cache_path(scene, frame_index)
            if os.path.exists(scene_frame_cache_path):
                return load_npy(scene_frame_cache_path)


        data_list: List[np.ndarray] = []

        # Pack batch input
        for bbox in bboxes:
            # Clip bbox to image
            bbox = copy.deepcopy(bbox)
            bbox.clip()
            assert bbox.area > 0, 'Bounding box has no area!'

            crop = bbox.crop(frame)
            data = self.preprocess(crop)
            data_list.append(data)

        if len(data_list) == 0:
            objects_features = np.empty(0, dtype=np.float32)
        else:
            data = np.stack(data_list, 0)

            if self._batch_inference:
                objects_features = self.inference(data)
            else:
                objects_features = np.concatenate([self.inference(data[i][np.newaxis]) for i in range(data.shape[0])], 0)

            objects_features = self.postprocess(objects_features)

        if self._cache_path is not None:
            # Store features to cache for faster future inference
            assert scene is not None, 'Scene must be specified in order to perform caching!'
            scene_frame_cache_path = self._get_scene_frame_cache_path(scene, frame_index)
            store_npy(objects_features, scene_frame_cache_path)

        return objects_features


    def _get_scene_frame_cache_path(self, scene: str, frame_index: int) -> str:
        """
        Args:
            scene: Scene name
            frame_index: Scene frame number

        Returns:
            Path where the ReId data is cached.
        """
        return os.path.join(self._cache_path, scene, f'reid_{frame_index:06d}.npy')
