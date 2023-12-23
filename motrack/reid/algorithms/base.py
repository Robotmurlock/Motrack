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
    def __init__(self, cache_path: Optional[str] = None):
        """
        Args:
            cache_path: Inference cache path
        """
        self._cache_path = cache_path

    @abstractmethod
    def extract_features(self, frame: np.ndarray, frame_index: int, scene: Optional[str] = None) -> np.ndarray:
        """
        Extracts object appearance features for given frame.

        Args:
            frame: Raw frame
            frame_index: Frame number (index)
            scene: Scene name (optional, for caching)

        Returns:
            Object frame appearance features
        """

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


        object_feature_list: List[np.ndarray] = []

        for bbox in bboxes:
            # Clip bbox to image
            bbox = copy.deepcopy(bbox)
            bbox.clip()
            assert bbox.area > 0, 'Bounding box has no area!'

            crop = bbox.crop(frame)
            object_features = self.extract_features(crop, frame_index=frame_index, scene=scene)
            object_feature_list.append(object_features)

        objects_features = np.concatenate(object_feature_list, 0) if len(object_feature_list) > 0 \
            else np.empty(0, dtype=np.float32)
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
