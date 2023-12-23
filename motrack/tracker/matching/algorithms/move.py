"""
Implementation of Move (hybrid) association method.
"""
from typing import Optional, List

import numpy as np

from motrack.library.cv.bbox import PredBBox
from motrack.tracker.matching.algorithms.iou import IoUAssociation, LabelGatingType
from motrack.tracker.matching.catalog import ASSOCIATION_CATALOG
from motrack.tracker.tracklet import Tracklet


def distance(name: str, x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculates distance between two vectors. Supports: L1 and L2.

    Args:
        name: Distance name
        x: Vector X
        y: Vector Y

    Returns:
        Distance between x and y.
    """
    if name == 'l1':
        return np.abs(x - y).sum()
    if name == 'l2':
        return np.sqrt(np.square(x - y).sum())
    if name == 'cosine':
        return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

    raise AssertionError('Invalid Program State!')


@ASSOCIATION_CATALOG.register('move')
class Move(IoUAssociation):
    """
    Combines Hungarian IOU and motion estimation for tracklet and detection matching.
    """
    DISTANCE_OPTIONS = ['l1', 'l2']
    def __init__(
        self,
        match_threshold: float = 0.30,
        motion_lambda: float = 5,
        distance_name: str = 'l1',
        label_gating: Optional[LabelGatingType] = None,
        fuse_score: bool = False,
        fast_linear_assignment: bool = False
    ):
        """
        Args:
            match_threshold: IOU match gating
            motion_lambda: Motion difference multiplier
            label_gating: Gating between different types of objects
            fuse_score: Fuse Hungarian IoU score
            fast_linear_assignment: Use greedy algorithm for linear assignment
                - This might be more efficient in case of large cost matrix
        """
        super().__init__(
            match_threshold=match_threshold,
            label_gating=label_gating,
            fuse_score=fuse_score,
            fast_linear_assignment=fast_linear_assignment
        )
        self._motion_lambda = motion_lambda

        assert distance_name in self.DISTANCE_OPTIONS, f'Invalid distance option "{distance_name}". Available: {self.DISTANCE_OPTIONS}.'
        self._distance_name = distance_name

    def _form_motion_distance_cost_matrix(
        self,
        tracklet_estimations: List[PredBBox],
        detections: List[PredBBox],
        tracklets: Optional[List[Tracklet]] = None
    ) -> np.ndarray:
        """
        Forms motion matrix where motion is approximated as the difference
        between bbox and estimated tracklet motion.

        Args:
            tracklet_estimations: Tracklet bbox estimations
            detections: Detection bboxes
            tracklets: Tracklets full info (history, etc.)

        Returns:
            Motion distance cost matrix
        """
        n_tracklets, n_detections = len(tracklet_estimations), len(detections)
        cost_matrix = np.zeros(shape=(n_tracklets, n_detections), dtype=np.float32)
        for t_i in range(n_tracklets):
            tracklet_estimated_bbox = tracklet_estimations[t_i]
            tracklet_info = tracklets[t_i]

            tracklet_last_bbox = tracklet_info.bbox.as_numpy_xyxy()
            tracklet_motion = tracklet_estimated_bbox.as_numpy_xyxy() - tracklet_last_bbox

            for d_i in range(n_detections):
                det_bbox = detections[d_i]
                det_motion = det_bbox.as_numpy_xyxy() - tracklet_last_bbox
                cost_matrix[t_i][d_i] = distance(self._distance_name, tracklet_motion, det_motion)

        return cost_matrix

    def _form_cost_matrix(
        self,
        tracklet_estimations: List[PredBBox],
        detections: List[PredBBox],
        object_features: Optional[np.ndarray] = None,
        tracklets: Optional[List[Tracklet]] = None
    ) -> np.ndarray:
        _ = object_features  # Unused

        neg_iou_cost_matrix = self._form_iou_cost_matrix(tracklet_estimations, detections)
        motion_diff_cost_matrix = self._form_motion_distance_cost_matrix(tracklet_estimations, detections, tracklets)
        return neg_iou_cost_matrix + self._motion_lambda * motion_diff_cost_matrix
