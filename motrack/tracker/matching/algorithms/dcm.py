"""
Implementation of SparseTrack DCM association method.
"""
from typing import Optional, List, Tuple

import numpy as np

from motrack.library.cv.bbox import PredBBox
from motrack.tracker.matching.algorithms.base import AssociationAlgorithm
from motrack.tracker.matching.algorithms.iou import IoUAssociation, LabelGatingType
from motrack.tracker.matching.algorithms.move import Move
from motrack.tracker.tracklet import Tracklet
from motrack.tracker.matching.catalog import ASSOCIATION_CATALOG


class DCM(AssociationAlgorithm):
    """
    SparseTrack DCM: https://arxiv.org/pdf/2306.05238.pdf
    """
    def __init__(
        self,
        matcher: AssociationAlgorithm,
        levels: int = 12,
        fast_linear_assignment: bool = False
    ):
        """
        Args:
            matcher: Base matcher
            levels: Number of depth levels to use
            fast_linear_assignment: Use greedy algorithm for linear assignment
                - This might be more efficient in case of large cost matrix
        """
        super().__init__(fast_linear_assignment=fast_linear_assignment)

        self._matcher = matcher
        self._levels = levels

    def _create_depth_masks(self, bboxes: List[PredBBox]) -> np.ndarray:
        """
        Arranges list of bboxes by levels.

        Args:
            bboxes: List of tracklet or detection bboxes

        Returns:
            Depth level class for each bbox
        """
        n_bboxes = len(bboxes)
        masks = np.full(shape=(self._levels, n_bboxes), fill_value=True, dtype=bool)
        if n_bboxes == 0:
            return masks

        neg_bottoms = [-bbox.bottom_right.y for bbox in bboxes]  # Pedestrian bbox bottom line
        min_bottom, max_bottom = min(neg_bottoms), max(neg_bottoms)
        depths_range = np.linspace(min_bottom, max_bottom, self._levels + 1, endpoint=True)

        prev_depth = depths_range[0]
        for i, depth in enumerate(depths_range[1:]):
            depth: float

            for j in range(n_bboxes):
                if i + 2 < depths_range.shape[0]:
                    masks[i, j] = prev_depth <= neg_bottoms[j] < depth
                else:
                    masks[i, j] = prev_depth <= neg_bottoms[j] <= depth

            prev_depth = depth

        assert masks.any(axis=0).all(), f'Invalid Mask! Info:\n{masks=}\n{masks.any(axis=1)=}\n{depths_range=}\n{neg_bottoms=}\n{bboxes}'
        return masks

    def match(
        self,
        tracklet_estimations: List[PredBBox],
        detections: List[PredBBox],
        object_features: Optional[np.ndarray] = None,
        tracklets: Optional[List[Tracklet]] = None
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        _ = object_features  # Unused

        t_masks = self._create_depth_masks(tracklet_estimations)
        d_masks = self._create_depth_masks(detections)

        unmatched_tracklet_indices: List[int] = []
        unmatched_detection_indices: List[int] = []
        matches: List[Tuple[int, int]] = []
        for tm, dm in zip(t_masks, d_masks):
            # Combine unmatched tracklets and detections with current DCM mask level
            level_tracklet_indices = [t_i for t_i, include in enumerate(tm) if include] + unmatched_tracklet_indices
            level_detection_indices = [d_i for d_i, include in enumerate(dm) if include] + unmatched_detection_indices

            # Extract info
            level_tracklet_estimations = [tracklet_estimations[t_i] for t_i in level_tracklet_indices]
            level_tracklets = [tracklets[t_i] for t_i in level_tracklet_indices] if tracklets is not None else None
            level_detections = [detections[d_i] for d_i in level_detection_indices]

            # Perform matching
            level_matches, unmatched_tracklet_indices, unmatched_detection_indices = self._matcher(
                level_tracklet_estimations, level_detections, tracklets=level_tracklets
            )

            # Map "level" indices to real indices
            level_matches = [(level_tracklet_indices[t_i], level_detection_indices[d_i]) for t_i, d_i in level_matches]
            unmatched_tracklet_indices = [level_tracklet_indices[t_i] for t_i in unmatched_tracklet_indices]
            unmatched_detection_indices = [level_detection_indices[d_i] for d_i in unmatched_detection_indices]

            matches.extend(level_matches)

        return matches, unmatched_tracklet_indices, unmatched_detection_indices


@ASSOCIATION_CATALOG.register('dcm')
class DCMIoU(DCM):
    """
    SparseTrack DCM (original): https://arxiv.org/pdf/2306.05238.pdf
    """
    def __init__(
        self,
        levels: int = 12,
        match_threshold: float = 0.30,
        fuse_score: bool = False,
        label_gating: Optional[LabelGatingType] = None,
        fast_linear_assignment: bool = False
    ):
        """
        Args:
            match_threshold: IoU gating match threshold
            fuse_score: Multiply IoU score by detection score
            levels: Number of depth levels to use
            fast_linear_assignment: Use greedy algorithm for linear assignment
                - This might be more efficient in case of large cost matrix
        """
        matcher = IoUAssociation(
            match_threshold=match_threshold,
            label_gating=label_gating,
            fuse_score=fuse_score
        )
        super().__init__(
            matcher=matcher,
            levels=levels,
            fast_linear_assignment=fast_linear_assignment
        )


@ASSOCIATION_CATALOG.register('move-dcm')
class MoveDCM(DCM):
    """
    DCM + Move
    """
    def __init__(
        self,
        levels: int = 12,
        match_threshold: float = 0.30,
        motion_lambda: float = 5,
        distance_name: str = 'l1',
        label_gating: Optional[LabelGatingType] = None,
        fuse_score: bool = False,
        fast_linear_assignment: bool = False
    ):
        """
        Args:
            levels: Number of depth levels to use
            match_threshold: IoU gating match threshold
            motion_lambda: Move cost multiplier
            fuse_score: Multiply IoU score by detection score
            levels: Number of depth levels to use
            fast_linear_assignment: Use greedy algorithm for linear assignment
                - This might be more efficient in case of large cost matrix
        """
        matcher = Move(
            match_threshold=match_threshold,
            motion_lambda=motion_lambda,
            distance_name=distance_name,
            label_gating=label_gating,
            fuse_score=fuse_score,
            fast_linear_assignment=fast_linear_assignment
        )
        super().__init__(
            matcher=matcher,
            levels=levels,
        )
