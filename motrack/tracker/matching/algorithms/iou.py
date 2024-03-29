"""
Standard SORT association method based solely on IoU.
"""
from typing import Optional, Union, List, Tuple

import numpy as np

from motrack.library.cv.bbox import PredBBox, BBox
from motrack.tracker.matching.algorithms.base import AssociationAlgorithm
from motrack.tracker.matching.catalog import ASSOCIATION_CATALOG
from motrack.tracker.tracklet import Tracklet, TrackletState

LabelType = Union[int, str]
LabelGatingType = Union[LabelType, List[Tuple[LabelType, LabelType]]]


class IoUBasedAssociation(AssociationAlgorithm):
    """
    Standard negative IoU association with gating.
    """

    def __init__(
        self,
        label_gating: Optional[LabelGatingType] = None,
        fast_linear_assignment: bool = False
    ):
        """
        Args:
            label_gating: Define which object labels can be matched
                - If not defined, matching is label agnostic
            fast_linear_assignment: Use greedy algorithm for linear assignment
                - This might be more efficient in case of large cost matrix
        """
        super().__init__(fast_linear_assignment=fast_linear_assignment)

        self._label_gating = {tuple(e) for e in label_gating} if label_gating is not None else None

    def _can_match_labels(self, tracklet_label: LabelType, det_label: LabelType) -> bool:
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
            tracklet = tracklets[t_i]
            tracklet_bbox = tracklet_estimations[t_i]

            for d_i in range(n_detections):
                det_bbox = detections[d_i]

                # Check if matching is possible
                if not self._can_match_labels(tracklet_bbox.label, det_bbox.label):
                    cost_matrix[t_i][d_i] = np.inf
                    continue

                cost_matrix[t_i][d_i] = self._calc_score(tracklet, tracklet_bbox, det_bbox)

        return cost_matrix

    def _calc_score(self, tracklet: Tracklet, tracklet_bbox: PredBBox, det_bbox: PredBBox) -> float:
        """
        Calculates matching score (cost) between tracklet and a detection bounding box.

        Args:
            tracklet: Tracklet info
            tracklet_bbox: Tracklet bounding box
            det_bbox: Detection bounding box

        Returns:
            Matching score
        """
        raise NotImplementedError('Score calculation is not implemented!')


@ASSOCIATION_CATALOG.register('iou')
class IoUAssociation(IoUBasedAssociation):
    """
    Standard negative IoU association with gating.
    """
    def __init__(
        self,
        match_threshold: float = 0.30,
        fuse_score: bool = False,
        label_gating: Optional[LabelGatingType] = None,
        fast_linear_assignment: bool = False
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
        super().__init__(
            label_gating=label_gating,
            fast_linear_assignment=fast_linear_assignment
        )

        self._match_threshold = match_threshold
        self._fuse_score = fuse_score

    def _calc_score(self, tracklet: Tracklet, tracklet_bbox: PredBBox, det_bbox: PredBBox) -> float:
        _ = tracklet

        # Calculate IOU score
        iou_score = tracklet_bbox.iou(det_bbox)
        # Higher the IOU score the better is the match (using negative values because of min optim function)
        # If score has very high value then
        score = iou_score * det_bbox.conf if self._fuse_score else iou_score
        return - score if score > self._match_threshold else np.inf


@ASSOCIATION_CATALOG.register('adaptive-iou')
class IoUAssociationAdaptiveGating(IoUBasedAssociation):
    """
    Adaptive IoU scales the match threshold based on the object's velocity.
    Object's that have less movement should have more overlap in the next frame.
    """
    def __init__(
        self,
        match_threshold: float = 0.30,
        fuse_score: bool = False,
        gating_factor: float = 1.0,
        label_gating: Optional[LabelGatingType] = None,
        fast_linear_assignment: bool = False
    ):
        """
        Args:
            match_threshold: Min threshold do match tracklet with object.
                If threshold is not met then cost is equal to infinity.
            fuse_score: Fuse score with iou
            gating_factor: Adaptive gating factor
            label_gating: Define which object labels can be matched
                - If not defined, matching is label agnostic
            fast_linear_assignment: Use greedy algorithm for linear assignment
                - This might be more efficient in case of large cost matrix
        """
        super().__init__(
            label_gating=label_gating,
            fast_linear_assignment=fast_linear_assignment
        )

        self._match_threshold = match_threshold
        self._fuse_score = fuse_score
        self._gating_factor = gating_factor

    def _calc_score(self, tracklet: Tracklet, tracklet_bbox: PredBBox, det_bbox: PredBBox) -> float:
        # Estimate variable gating threshold
        match_threshold = self._match_threshold
        if len(tracklet.history) >= 2:
            last_bbox_index, last_bbox = tracklet.history[-1]
            prev_bbox_index, prev_bbox = tracklet.history[-2]
            if last_bbox_index > prev_bbox_index:
                movement = np.abs(last_bbox.as_numpy_xyxy() - prev_bbox.as_numpy_xyxy() / (last_bbox_index - prev_bbox_index)).mean()
                match_threshold = max(0, match_threshold - movement * self._gating_factor)

        iou_score = tracklet_bbox.iou(det_bbox)
        score = iou_score * det_bbox.conf if self._fuse_score else iou_score
        return - score if score > match_threshold else np.inf


@ASSOCIATION_CATALOG.register('hmiou')
class HMIoUAssociation(IoUBasedAssociation):
    """
    Height modulated IoU. Reference: https://arxiv.org/pdf/2308.00783.pdf
    """
    def __init__(
        self,
        match_threshold: float = 0.30,
        label_gating: Optional[LabelGatingType] = None,
        fast_linear_assignment: bool = False
    ):
        """
        Args:
            match_threshold: Min threshold do match tracklet with object.
                If threshold is not met then cost is equal to infinity.
            label_gating: Define which object labels can be matched
                - If not defined, matching is label agnostic
            fast_linear_assignment: Use greedy algorithm for linear assignment
                - This might be more efficient in case of large cost matrix
        """
        super().__init__(
            label_gating=label_gating,
            fast_linear_assignment=fast_linear_assignment
        )

        self._match_threshold = match_threshold

    @staticmethod
    def hiou(tracklet_bbox: PredBBox, det_bbox: PredBBox) -> float:
        """
        Calculates HIoU between two bboxes.

        Args:
            tracklet_bbox: Tracklet bounding box
            det_bbox: Detection bounding box

        Returns:
            HIoU between two bboxes
        """
        intersection_bottom = min(tracklet_bbox.bottom_right.y, det_bbox.bottom_right.y)
        intersection_top = max(tracklet_bbox.upper_left.y, det_bbox.upper_left.y)
        union_bottom = max(tracklet_bbox.bottom_right.y, det_bbox.bottom_right.y)
        union_top = min(tracklet_bbox.upper_left.y, det_bbox.upper_left.y)

        if union_top >= union_bottom or intersection_top >= intersection_bottom:
            return 0.0

        return (intersection_bottom - intersection_top) / (union_bottom - union_top)

    def _calc_score(self, tracklet: Tracklet, tracklet_bbox: PredBBox, det_bbox: PredBBox) -> float:
        _ = tracklet

        # Calculate IOU score
        iou_score = tracklet_bbox.iou(det_bbox)
        height_iou_score = self.hiou(tracklet_bbox, det_bbox)
        return - iou_score * height_iou_score * height_iou_score if iou_score > self._match_threshold else np.inf


@ASSOCIATION_CATALOG.register('decay-iou')
class DecayIoU(IoUBasedAssociation):
    """
    IoU but with threshold decay.
    """
    def __init__(
        self,
        min_threshold: float = 0.30,
        max_threshold: float = 0.50,
        threshold_decay: float = 0.02,
        fuse_score: bool = False,
        label_gating: Optional[LabelGatingType] = None,
        expansion_rate: float = 0.0,
        fast_linear_assignment: bool = False
    ):
        """
        Args:
            min_threshold: Min IoU threshold
            max_threshold: Initial IoU threshold
            threshold_decay: IoU threshold decay
            fuse_score: Fuse score with iou
            label_gating: Define which object labels can be matched
                - If not defined, matching is label agnostic
            expansion_rate: Bounding box expansion rate
            fast_linear_assignment: Use greedy algorithm for linear assignment
                - This might be more efficient in case of large cost matrix
        """
        super().__init__(
            label_gating=label_gating,
            fast_linear_assignment=fast_linear_assignment
        )

        self._min_threshold = min_threshold
        self._max_threshold = max_threshold
        self._threshold_decay = threshold_decay
        self._fuse_score = fuse_score

        assert expansion_rate >= 0.0, f'Invalid value for expansion rate: {expansion_rate}.'
        self._expansion_rate = expansion_rate

    def _expand_bbox(self, bbox: PredBBox) -> PredBBox:
        """
        Expands bounding box (allows matching between bounding boxes that do not overlap).

        Args:
            bbox: Bounding box

        Returns:
            Expanded (new) bounding box
        """
        h = bbox.height * (1 + self._expansion_rate)
        w = bbox.width * (1 + self._expansion_rate)
        center = bbox.center
        cx, cy = center.x, center.y

        return PredBBox.create(
            bbox=BBox.from_cxywh(cx, cy, w, h),
            label=bbox.label,
            conf=bbox.conf
        )

    def _calc_score(self, tracklet: Tracklet, tracklet_bbox: PredBBox, det_bbox: PredBBox) -> float:
        _ = tracklet

        # Calculate IOU score
        if self._expansion_rate > 0.0:
            tracklet_bbox = self._expand_bbox(tracklet_bbox)
            det_bbox = self._expand_bbox(det_bbox)

        iou_score = tracklet_bbox.iou(det_bbox)
        # Higher the IOU score the better is the match (using negative values because of min optim function)
        # If score has very high value then
        score = iou_score * det_bbox.conf if self._fuse_score else iou_score

        match_threshold = max(self._min_threshold, self._max_threshold - self._threshold_decay * tracklet.lost_time)
        return - score if score > match_threshold else np.inf

