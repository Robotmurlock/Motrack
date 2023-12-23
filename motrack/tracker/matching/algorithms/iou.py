"""
Standard SORT association method based solely on IoU.
"""
from typing import Optional, Union, List, Tuple

import numpy as np

from motrack.library.cv.bbox import PredBBox
from motrack.tracker.matching.algorithms.base import AssociationAlgorithm
from motrack.tracker.matching.utils import hungarian, greedy
from motrack.tracker.tracklet import Tracklet
from motrack.tracker.matching.catalog import ASSOCIATION_CATALOG


LabelType = Union[int, str]
LabelGatingType = Union[LabelType, List[Tuple[LabelType, LabelType]]]


@ASSOCIATION_CATALOG.register('iou')
class IoUAssociation(AssociationAlgorithm):
    """
    Solves the linear sum assignment problem from given cost matrix based on IOU scores.
    """
    def __init__(
        self,
        match_threshold: float = 0.30,
        fuse_score: bool = False,
        label_gating: Optional[LabelGatingType] = None,
        fast_linear_assignment: bool = False,
        *args, **kwargs
    ):
        """
        Args:
            match_threshold: Min threshold do match tracklet with object.
                If threshold is not met then cost is equal to infinity.
            fuse_score: Fuse score with iou
            label_gating: Define which object labels can be matched
                - If not defined, matching is label agnostic
            fast_linear_assignment: Use greedy algorithm for linear assignment
                - This might be more efficient in case of large cost matrix
        """
        super().__init__(*args, **kwargs)

        self._match_threshold = match_threshold
        self._fuse_score = fuse_score

        self._label_gating = set([tuple(e) for e in label_gating]) if label_gating is not None else None
        self._fast_linear_assignment = fast_linear_assignment

    def _can_match(self, tracklet_label: LabelType, det_label: LabelType) -> bool:
        """
        Checks if matching between tracklet and detection is possible.

        Args:
            tracklet_label: Tracklet label
            det_label: Detection label

        Returns:
            True if matching is possible else False
        """
        if tracklet_label == det_label:
            # Objects with same label can always match
            return True

        if self._label_gating is None:
            # If label gating is not set then any objects with same label can't match
            return False

        return (tracklet_label, det_label) in self._label_gating \
            or (det_label, tracklet_label) in self._label_gating

    def _form_iou_cost_matrix(self, tracklet_estimations: List[PredBBox], detections: List[PredBBox]) -> np.ndarray:
        """
        Creates negative IOU cost matrix as an input into Hungarian algorithm.

        Args:
            tracklet_estimations: List of tracklet estimated bboxes
            detections: Detection (observation) bboxes

        Returns:
            Negative IOU cost matrix
        """
        n_tracklets, n_detections = len(tracklet_estimations), len(detections)
        cost_matrix = np.zeros(shape=(n_tracklets, n_detections), dtype=np.float32)
        for t_i in range(n_tracklets):
            tracklet_bbox = tracklet_estimations[t_i]

            for d_i in range(n_detections):
                det_bbox = detections[d_i]

                # Check if matching is possible
                if not self._can_match(tracklet_bbox.label, det_bbox.label):
                    cost_matrix[t_i][d_i] = np.inf
                    continue

                # Calculate IOU score
                iou_score = tracklet_bbox.iou(det_bbox)
                # Higher the IOU score the better is the match (using negative values because of min optim function)
                # If score has very high value then
                score = iou_score * det_bbox.conf if self._fuse_score else iou_score
                score = - score if score > self._match_threshold else np.inf
                cost_matrix[t_i][d_i] = score

        return cost_matrix

    def match(
        self,
        tracklet_estimations: List[PredBBox],
        detections: List[PredBBox],
        object_features: Optional[np.ndarray] = None,
        tracklets: Optional[List[Tracklet]] = None
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        _ = object_features  # Unused

        cost_matrix = self._form_iou_cost_matrix(tracklet_estimations, detections)
        if self._fast_linear_assignment:
            return greedy(cost_matrix)
        return hungarian(cost_matrix)
