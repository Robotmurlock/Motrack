"""
Implementation of Move (hybrid) association method.
"""
from typing import Optional, List

import numpy as np
from pydantic import BaseModel, ConfigDict

from motrack.library.cv.bbox import PredBBox
from motrack.tracker.matching.algorithms.base import AssociationAlgorithm
from motrack.tracker.matching.algorithms.compose import ComposeAssociationAlgorithm, ComposeAssociationConfig
from motrack.tracker.matching.algorithms.iou import IoUAssociation, IoUAssociationConfig, LabelGatingType
from motrack.tracker.matching.catalog import ASSOCIATION_CATALOG
from motrack.tracker.tracklet import Tracklet


@ASSOCIATION_CATALOG.register_config('distance')
class DistanceAssociationConfig(BaseModel):
    """
    Config for distance association.
    """

    model_config = ConfigDict(extra='forbid')

    distance_name: str = 'l1'
    fast_linear_assignment: bool = False


@ASSOCIATION_CATALOG.register_config('move')
class MoveAssociationConfig(IoUAssociationConfig):
    """
    Config for Move association.
    """

    motion_lambda: float = 5
    distance_name: str = 'l1'


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
    def __init__(self, config: DistanceAssociationConfig):
        """
        Args:
            distance_name: Distance function mae
            fast_linear_assignment: Use greedy algorithm for linear assignment
                - This might be more efficient in case of large cost matrix
        """
        super().__init__(fast_linear_assignment=config.fast_linear_assignment)
        self._distance_name = config.distance_name


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
    def __init__(self, config: MoveAssociationConfig):
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
                IoUAssociationConfig(
                    match_threshold=config.match_threshold,
                    label_gating=config.label_gating,
                    fuse_score=config.fuse_score,
                )
            ),
            DistanceAssociation(
                DistanceAssociationConfig(
                    distance_name=config.distance_name,
                )
            )
        ]
        weights = [1, config.motion_lambda]

        super().__init__(
            ComposeAssociationConfig(
                matchers=[],
                weights=weights,
                fast_linear_assignment=config.fast_linear_assignment,
            ),
            matchers=matchers,
        )
