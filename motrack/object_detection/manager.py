"""
DetectionManager makes tracker development easier and faster
by adding cache and oracle detections options.
"""
import os
from typing import Optional, List, Tuple, Union

import numpy as np

from motrack.datasets import BaseDataset
from motrack.library.cv.bbox import PredBBox
from motrack.library.numpy_utils.io import load_npy, store_npy
from motrack.object_detection.factory import object_detection_inference_factory
from motrack.utils.lookup import LookupTable


class DetectionManager:
    """
    Detection manager integrates multiple features, making the usage flexible. Features:
    - Performs inference
    - Uses cache for faster multiple runs
    - Uses optional `oracle` GT detections used for debugging and testing
    """
    def __init__(
        self,
        inference_name: str,
        inference_params: dict,
        dataset: BaseDataset,
        lookup: Optional[LookupTable] = None,
        cache_path: Optional[str] = None,
        oracle: bool = False
    ):
        """
        Args:
            inference_name: Inference model name
            inference_params: Inference model parameters
            dataset: Dataset
            lookup: Lookup (if set then returns string labels instead of ints)
            cache_path: If set then cache is used
            oracle: Use GT predictions instead of running inference
        """
        self._inference = object_detection_inference_factory(name=inference_name, params=inference_params, lookup=lookup)
        self._dataset = dataset
        self._lookup = lookup
        self._cache_path = cache_path
        self._oracle = oracle

    def predict(self, scene: str, frame_index: int) -> List[PredBBox]:
        """
        Performs prediction for given scene name and frame index.

        If `cache_path` is set then inference is faster after first run.
        If `oracle` is set to True then ground truth is used instead of inference

        Args:
            scene: Scene name
            frame_index: Frame inde

        Returns:
            List of inference bboxes
        """
        if self._oracle:
            bboxes, classes, confidences = self._oracle_inference(scene, frame_index)
        else:
            scene_frame_cache_path, _, _, _ = self._get_cache_paths(scene, frame_index)
            if self._cache_path is not None and os.path.exists(scene_frame_cache_path):
                bboxes, classes, confidences = self._load_from_cache(scene, frame_index)
            else:
                image = self._dataset.load_scene_image_by_frame_index(scene, frame_index)
                bboxes, classes, confidences = self._inference.predict_with_postprocess(image)
                if self._cache_path is not None:
                    self._store_cache(scene, frame_index, bboxes, classes, confidences)

        return self._inference.pack_bboxes(bboxes, classes, confidences)

    def _load_from_cache(self, scene: str, frame_index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads inference info (bboxes, classes, confidences) from cache.

        Args:
            scene: Scene name
            frame_index: Frame index

        Returns:
            Cached inference data
        """
        _, bboxes_cache_path, classes_cache_path, confidences_cache_path = self._get_cache_paths(scene, frame_index)
        return load_npy(bboxes_cache_path), load_npy(classes_cache_path), load_npy(confidences_cache_path)

    def _store_cache(self, scene: str, frame_index: int, bboxes: np.ndarray, classes: np.ndarray, confidences: np.ndarray) -> None:
        """
        Stores cache for faster inference in next iteration

        Args:
            scene: Scene name
            frame_index: Frame index
            bboxes: BBoxes data (numpy)
            classes: Classes data (numpy)
            confidences: Confidences data (numpy)
        """
        _, bboxes_cache_path, classes_cache_path, confidences_cache_path = self._get_cache_paths(scene, frame_index)
        store_npy(bboxes, bboxes_cache_path)
        store_npy(classes, classes_cache_path)
        store_npy(confidences, confidences_cache_path)

    def _get_cache_paths(self, scene: str, frame_index: int) -> Tuple[str, str, str, str]:
        """
        Get path where the inference cache can be found

        Args:
            scene: Scene name
            frame_index: Frame index

        Returns:
            Scene frame cache path
        """
        scene_frame_cache_path = os.path.join(self._cache_path, scene, f'{frame_index:06d}')
        bboxes_cache_path = os.path.join(scene_frame_cache_path, 'bboxes.npy')
        classes_cache_path = os.path.join(scene_frame_cache_path, 'classes.npy')
        confidences_cache_path = os.path.join(scene_frame_cache_path, 'confidences.npy')
        return scene_frame_cache_path, bboxes_cache_path, classes_cache_path, confidences_cache_path

    def _oracle_inference(self, scene: str, frame_index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns GT prediction (annotations)

        Args:
            scene: Scene name
            frame_index: Frame index

        Returns:
            Oracle detections
        """
        object_ids = self._dataset.get_scene_object_ids(scene)
        frame_bboxes: List[np.ndarray] = []
        frame_classes: List[Union[int, float]] = []
        frame_confidences: List[float] = []

        for object_id in object_ids:
            data = self._dataset.get_object_data_by_frame_index(object_id, frame_index, relative_bbox_coords=True)
            if data is None:
                continue

            bbox = data.bbox
            if isinstance(bbox, list):
                bbox = np.array(bbox, dtype=np.float32)
            bbox[[2, 3]] += bbox[[0, 1]]
            cls = self._lookup[data.category]

            frame_bboxes.append(bbox)
            frame_classes.append(cls)
            frame_confidences.append(1.0)

        bboxes = np.stack(frame_bboxes)
        classes = np.array(frame_classes, dtype=np.float32)
        confidences = np.array(frame_confidences, dtype=np.float32)
        return bboxes, classes, confidences
