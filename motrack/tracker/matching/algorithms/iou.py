"""
Standard SORT association method based solely on IoU.
"""
from typing import Optional, Union, List, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from motrack.library.cv.bbox import PredBBox, BBox
from motrack.tracker.matching.algorithms.base import AssociationAlgorithm
from motrack.tracker.matching.catalog import ASSOCIATION_CATALOG
from motrack.tracker.tracklet import Tracklet, TrackletState

LabelType = Union[int, str]
LabelGatingType = Union[LabelType, List[Tuple[LabelType, LabelType]]]


class IoUBasedAssociationConfig(BaseModel):
    """
    Shared IoU-based association config.
    """

    model_config = ConfigDict(extra='forbid')

    label_gating: Optional[LabelGatingType] = None
    fast_linear_assignment: bool = False


@ASSOCIATION_CATALOG.register_config('iou')
class IoUAssociationConfig(IoUBasedAssociationConfig):
    """
    Config for IoU association.
    """

    match_threshold: float = 0.30
    fuse_score: bool = False


@ASSOCIATION_CATALOG.register_config('adaptive-iou')
class AdaptiveIoUAssociationConfig(IoUAssociationConfig):
    """
    Config for adaptive IoU association.
    """

    gating_factor: float = 1.0


@ASSOCIATION_CATALOG.register_config('hmiou')
class HMIoUAssociationConfig(IoUBasedAssociationConfig):
    """
    Config for HMIoU association.
    """

    match_threshold: float = 0.30


@ASSOCIATION_CATALOG.register_config('decay-iou')
class DecayIoUAssociationConfig(IoUBasedAssociationConfig):
    """
    Config for decay IoU association.
    """

    min_threshold: float = 0.30
    max_threshold: float = 0.50
    threshold_decay: float = 0.02
    fuse_score: bool = False
    expansion_rate: float = Field(default=0.0, ge=0.0)


class IoUBasedAssociation(AssociationAlgorithm):
    """
    Standard negative IoU association with gating.
    """

    def __init__(self, config: IoUBasedAssociationConfig):
        """
        Args:
            label_gating: Define which object labels can be matched
                - If not defined, matching is label agnostic
            fast_linear_assignment: Use greedy algorithm for linear assignment
                - This might be more efficient in case of large cost matrix
        """
        super().__init__(fast_linear_assignment=config.fast_linear_assignment)

        label_gating = config.label_gating
        self._label_gating = {tuple(e) for e in label_gating} if label_gating is not None and isinstance(label_gating, list) else None

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
        if n_tracklets == 0 or n_detections == 0:
            return np.zeros(shape=(n_tracklets, n_detections), dtype=np.float32)

        t_xyxy = np.array([t.as_numpy_xyxy() for t in tracklet_estimations])
        d_xyxy = np.array([d.as_numpy_xyxy() for d in detections])
        iou_matrix = BBox.batch_iou(t_xyxy, d_xyxy)

        cost_matrix = self._calc_score_matrix(iou_matrix, tracklets, tracklet_estimations, detections)

        # Apply label gating
        for t_i in range(n_tracklets):
            for d_i in range(n_detections):
                if not self._can_match_labels(tracklet_estimations[t_i].label, detections[d_i].label):
                    cost_matrix[t_i][d_i] = np.inf

        return cost_matrix

    def _calc_score_matrix(
        self,
        iou_matrix: np.ndarray,
        tracklets: Optional[List[Tracklet]],
        tracklet_estimations: List[PredBBox],
        detections: List[PredBBox],
    ) -> np.ndarray:
        """
        Compute cost matrix from a precomputed IoU matrix (vectorized path).
        Subclasses override this for vectorized scoring.
        Falls back to per-element _calc_score() by default.
        """
        n_tracklets, n_detections = iou_matrix.shape
        cost_matrix = np.zeros(shape=(n_tracklets, n_detections), dtype=np.float32)
        for t_i in range(n_tracklets):
            for d_i in range(n_detections):
                cost_matrix[t_i][d_i] = self._calc_score(tracklets[t_i], tracklet_estimations[t_i], detections[d_i])
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
    def __init__(self, config: IoUAssociationConfig):
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
        super().__init__(config)

        self._match_threshold = config.match_threshold
        self._fuse_score = config.fuse_score

    def _calc_score_matrix(self, iou_matrix, tracklets, tracklet_estimations, detections):
        if self._fuse_score:
            confs = np.array([d.conf for d in detections], dtype=np.float32)
            score_matrix = iou_matrix * confs[None, :]
        else:
            score_matrix = iou_matrix.copy()

        cost_matrix = -score_matrix
        cost_matrix[score_matrix <= self._match_threshold] = np.inf
        return cost_matrix

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
    def __init__(self, config: AdaptiveIoUAssociationConfig):
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
        super().__init__(config)

        self._match_threshold = config.match_threshold
        self._fuse_score = config.fuse_score
        self._gating_factor = config.gating_factor

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
    def __init__(self, config: HMIoUAssociationConfig):
        """
        Args:
            match_threshold: Min threshold do match tracklet with object.
                If threshold is not met then cost is equal to infinity.
            label_gating: Define which object labels can be matched
                - If not defined, matching is label agnostic
            fast_linear_assignment: Use greedy algorithm for linear assignment
                - This might be more efficient in case of large cost matrix
        """
        super().__init__(config)

        self._match_threshold = config.match_threshold

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

    def _calc_score_matrix(self, iou_matrix, tracklets, tracklet_estimations, detections):
        t_xyxy = np.array([t.as_numpy_xyxy() for t in tracklet_estimations])
        d_xyxy = np.array([d.as_numpy_xyxy() for d in detections])

        # Vectorized height IoU
        int_bottom = np.minimum(t_xyxy[:, 3:4], d_xyxy[:, 3:4].T)
        int_top = np.maximum(t_xyxy[:, 1:2], d_xyxy[:, 1:2].T)
        union_bottom = np.maximum(t_xyxy[:, 3:4], d_xyxy[:, 3:4].T)
        union_top = np.minimum(t_xyxy[:, 1:2], d_xyxy[:, 1:2].T)

        valid = (union_top < union_bottom) & (int_top < int_bottom)
        hiou = np.where(valid, (int_bottom - int_top) / np.maximum(union_bottom - union_top, 1e-8), 0.0)

        cost_matrix = -(iou_matrix * hiou * hiou)
        cost_matrix[iou_matrix <= self._match_threshold] = np.inf
        return cost_matrix

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
    def __init__(self, config: DecayIoUAssociationConfig):
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
        super().__init__(config)

        self._min_threshold = config.min_threshold
        self._max_threshold = config.max_threshold
        self._threshold_decay = config.threshold_decay
        self._fuse_score = config.fuse_score
        self._expansion_rate = config.expansion_rate

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

    def _calc_score_matrix(self, iou_matrix, tracklets, tracklet_estimations, detections):
        if self._expansion_rate > 0.0:
            # Recompute IoU with expanded bboxes
            t_xyxy = np.array([t.as_numpy_xyxy() for t in tracklet_estimations])
            d_xyxy = np.array([d.as_numpy_xyxy() for d in detections])

            # Expand: increase w and h by expansion_rate, keep center
            def expand_xyxy(coords):
                cx = (coords[:, 0] + coords[:, 2]) / 2
                cy = (coords[:, 1] + coords[:, 3]) / 2
                w = (coords[:, 2] - coords[:, 0]) * (1 + self._expansion_rate)
                h = (coords[:, 3] - coords[:, 1]) * (1 + self._expansion_rate)
                return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

            iou_matrix = BBox.batch_iou(expand_xyxy(t_xyxy), expand_xyxy(d_xyxy))

        if self._fuse_score:
            confs = np.array([d.conf for d in detections], dtype=np.float32)
            score_matrix = iou_matrix * confs[None, :]
        else:
            score_matrix = iou_matrix.copy()

        # Per-tracklet threshold based on lost_time
        lost_times = np.array([t.lost_time for t in tracklets], dtype=np.float32)
        thresholds = np.maximum(self._min_threshold, self._max_threshold - self._threshold_decay * lost_times)

        cost_matrix = -score_matrix
        cost_matrix[score_matrix <= thresholds[:, None]] = np.inf
        return cost_matrix

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

