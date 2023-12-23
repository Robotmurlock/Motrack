"""
Implementation of C-BIoU and BIoU association methods.
"""
from typing import Optional, List, Tuple

import numpy as np

from motrack.library.cv.bbox import PredBBox, BBox
from motrack.tracker.matching.algorithms.base import AssociationAlgorithm
from motrack.tracker.matching.algorithms.iou import IoUAssociation, LabelGatingType
from motrack.tracker.matching.catalog import ASSOCIATION_CATALOG
from motrack.tracker.tracklet import Tracklet


@ASSOCIATION_CATALOG.register('biou')
class HungarianBIoU(IoUAssociation):
    """
    BIoU matching algorithm. Ref: https://arxiv.org/pdf/2211.14317.pdf
    """
    def __init__(
        self,
        b: float = 0.3,
        match_threshold: float = 0.30,
        label_gating: Optional[LabelGatingType] = None,
        *args, **kwargs
    ):
        """
        Args:
            b: BBox buffer
            match_threshold: IOU match gating
            label_gating: Gating between different types of objects
        """
        super().__init__(
            match_threshold=match_threshold,
            label_gating=label_gating,
            *args, **kwargs
        )
        self._b = b

    def _buffer_bbox(self, bbox: PredBBox) -> PredBBox:
        center = bbox.center
        w, h = bbox.width, bbox.height
        w *= (1 + 2 * self._b)
        h *= (1 + 2 * self._b)

        return PredBBox.create(
            bbox=BBox.from_cxywh(center.x, center.y, w, h),
            label=bbox.label,
            conf=bbox.conf
        )

    def form_iou_cost_matrix(
        self,
        tracklet_estimations: List[PredBBox],
        detections: List[PredBBox],
        object_features: Optional[np.ndarray] = None,
        tracklets: Optional[List[Tracklet]] = None
    ) -> np.ndarray:
        _, _ = object_features, tracklets  # Unused

        tracklet_estimations = [self._buffer_bbox(bbox) for bbox in tracklet_estimations]
        detections = [self._buffer_bbox(bbox) for bbox in detections]
        return self.form_iou_cost_matrix(tracklet_estimations, detections)


@ASSOCIATION_CATALOG.register('cbiou')
class HungarianCBIoU(AssociationAlgorithm):
    """
    C-BIoU matching algorithm. Ref: https://arxiv.org/pdf/2211.14317.pdf
    """
    def __init__(
        self,
        b1: float = 0.3,
        b2: float = 0.4,
        match_threshold: float = 0.30,
        label_gating: Optional[LabelGatingType] = None,
        fast_linear_assignment: bool = False,
    ):
        """
        Args:
            b1: First buffer matching threshold
            b2: Second buffer matching threshold
            match_threshold: IoU match gating
            label_gating: Gating between different types of objects
            fast_linear_assignment: Use greedy algorithm for linear assignment
            - This might be more efficient in case of large cost matrix
        """
        super().__init__(fast_linear_assignment=fast_linear_assignment)

        self._biou1_matcher = HungarianBIoU(
            b=b1,
            match_threshold=match_threshold,
            label_gating=label_gating,
        )

        self._biou1_matcher = HungarianBIoU(
            b=b2,
            match_threshold=match_threshold,
            label_gating=label_gating,
        )

    def match(
        self,
        tracklet_estimations: List[PredBBox],
        detections: List[PredBBox],
        object_features: Optional[np.ndarray] = None,
        tracklets: Optional[List[Tracklet]] = None
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        _ = object_features  # Unused

        # First matching
        matches1, unmatched_tracklet_indices1, unmatched_detection_indices1 = \
            self._biou1_matcher(tracklet_estimations, detections, tracklets=tracklets)
        unmatched_tracklet_estimations = [tracklet_estimations[t_i] for t_i in unmatched_tracklet_indices1]
        unmatched_tracklets = [tracklets[t_i] for t_i in unmatched_tracklet_indices1]
        unmatched_detections = [detections[d_i] for d_i in unmatched_detection_indices1]

        # Second matching
        matches2, unmatched_tracklet_indices2, unmatched_detection_indices2 = \
            self._biou1_matcher(unmatched_tracklet_estimations, unmatched_detections, tracklets=unmatched_tracklets)
        unmatched_tracklet_indices = [unmatched_tracklet_indices1[t_i] for t_i in unmatched_tracklet_indices2]
        unmatched_detection_indices = [unmatched_detection_indices1[t_i] for t_i in unmatched_detection_indices2]
        matches2 = [(unmatched_tracklet_indices1[t_i], unmatched_detection_indices1[d_i]) for t_i, d_i in matches2]
        matches = matches1 + matches2

        return matches, unmatched_tracklet_indices, unmatched_detection_indices
