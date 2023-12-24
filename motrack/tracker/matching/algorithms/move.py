"""
Implementation of Move (hybrid) association method.
"""
from typing import Optional, List

import numpy as np

from motrack.library.cv.bbox import PredBBox
from motrack.tracker.matching.algorithms.base import AssociationAlgorithm
from motrack.tracker.matching.algorithms.compose import ComposeAssociationAlgorithm
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

    raise AssertionError('Invalid Program State!')


@ASSOCIATION_CATALOG.register('distance')
class DistanceAssociation(AssociationAlgorithm):
    """
    Calculates distance between detection and prediction relative to the last track position.
    """
    def __init__(
        self,
        distance_name: str = 'l1',
        fast_linear_assignment: bool = False
    ):
        """
        Args:
            distance_name: Distance function mae
            fast_linear_assignment: Use greedy algorithm for linear assignment
                - This might be more efficient in case of large cost matrix
        """
        super().__init__(fast_linear_assignment=fast_linear_assignment)
        self._distance_name = distance_name


    def form_cost_matrix(
        self,
        tracklet_estimations: List[PredBBox],
        detections: List[PredBBox],
        object_features: Optional[np.ndarray] = None,
        tracklets: Optional[List[Tracklet]] = None
    ) -> np.ndarray:
        _ = object_features  # Unused

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


@ASSOCIATION_CATALOG.register('move')
class Move(ComposeAssociationAlgorithm):
    """
    Combines Hungarian IOU and motion estimation for tracklet and detection matching.
    """
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
        matchers = [
            IoUAssociation(
                match_threshold=match_threshold,
                label_gating=label_gating,
                fuse_score=fuse_score
            ),
            DistanceAssociation(
                distance_name=distance_name
            )
        ]
        weights = [1, motion_lambda]

        super().__init__(
            matchers=matchers,
            weights=weights,
            fast_linear_assignment=fast_linear_assignment
        )
