"""
Standard DeepSORT association method based on IoU and appearance info.
"""
from typing import List, Tuple
from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist

from motrack.library.cv import PredBBox
from motrack.tracker.matching.algorithms.iou import LabelGatingType, IoUAssociation
from motrack.tracker.matching.catalog import ASSOCIATION_CATALOG
from motrack.tracker.matching.utils import hungarian, greedy
from motrack.tracker.tracklet import TrackletCommonData, Tracklet


@ASSOCIATION_CATALOG.register('reid-iou')
class ReidIoUAssociation(IoUAssociation):
    """
    Standard DeepSORT association method based on IoU and appearance.
    """
    def __init__(
        self,
        appearance_weight: float = 0.98,
        appearance_threshold: float = 0.5,
        appearance_metric: str = 'cosine',
        appearance_ema_momentum: float = 0.9,
        match_threshold: float = 0.30,
        fuse_score: bool = False,
        label_gating: Optional[LabelGatingType] = None,
        fast_linear_assignment: bool = False,
        *args, **kwargs
    ):
        """
        Args:
            appearance_weight: Appearance to IoU association weight
            appearance_threshold: Appearance metric minimum threshold
            appearance_metric: Appearance metric (default: cosine)
            appearance_ema_momentum: Appearance metric exponential moving average momentum
            match_threshold: IoU match threshold
            fuse_score: Fuse detection score with IoU match
            label_gating: Allow different labels to be matched
            fast_linear_assignment: Use faster linear assignment
        """
        assert 0 <= appearance_ema_momentum <= 1.0, '"ema_weight" must be in interval [0, 1]!'
        assert 0 <= appearance_weight <= 1.0, '"appearance_weight" must be in interval [0, 1]!'

        super().__init__(
            match_threshold=match_threshold,
            fuse_score=fuse_score,
            label_gating=label_gating,
            fast_linear_assignment=fast_linear_assignment,
            *args, **kwargs
        )

        self._appearance_weight = appearance_weight
        self._appearance_threshold = appearance_threshold
        self._appearance_metric: str = appearance_metric
        self._appearance_ema_momentum = appearance_ema_momentum

    def match(
        self,
        tracklet_estimations: List[PredBBox],
        detections: List[PredBBox],
        object_features: Optional[np.ndarray] = None,
        tracklets: Optional[List[Tracklet]] = None
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        assert tracklets is not None, 'Tracklets are mandatory for ReID based association!'
        assert object_features is not None, 'Detection features are also mandatory!'

        motion_cost_matrix = self._form_iou_cost_matrix(
            tracklet_estimations=tracklet_estimations,
            detections=detections
        )

        appearance_cost_matrix = self._form_appearance_cost_matrix(
            tracklets=tracklets,
            object_features=object_features
        )

        # Calculate cost matrix
        if self._appearance_weight == 0.0:
            cost_matrix = motion_cost_matrix  # This way we avoid 0.0 * inf = NaN edge-case
        elif self._appearance_weight == 1.0:
            cost_matrix = appearance_cost_matrix
        else:
            cost_matrix = (1 - self._appearance_weight) * motion_cost_matrix + self._appearance_weight * appearance_cost_matrix

        matches, unmatched_tracklets, unmatched_detections = greedy(cost_matrix) if self._fast_linear_assignment else \
            hungarian(cost_matrix)

        self._update_appearances(tracklets, object_features, matches)

        return matches, unmatched_tracklets, unmatched_detections


    def _form_appearance_cost_matrix(self, tracklets: List[Tracklet], object_features: np.ndarray) -> np.ndarray:
        """
        Forms appearance cost matrix.

        Args:
            tracklets: List of tracklets
            object_features: Object features for new detected objects

        Returns:
            Appearance cost matrix
        """
        n_detections = object_features.shape[0]
        n_tracklets = len(tracklets)

        if n_tracklets == 0 or n_detections == 0:
            return np.zeros(shape=(n_tracklets, n_detections), dtype=np.float32)

        tracklet_features = np.stack([t.get(TrackletCommonData.APPEARANCE) for t in tracklets])
        # noinspection PyTypeChecker
        cost_matrix = np.maximum(0.0, cdist(tracklet_features, object_features, metric=self._appearance_metric))
        cost_matrix[cost_matrix > 1 - self._appearance_threshold] = np.inf
        return cost_matrix


    def _update_appearances(self, tracklets: List[Tracklet], object_features: np.ndarray, matches: List[Tuple[int, int]]) -> None:
        """
        Updates Tracklet appearance.

        Args:
            tracklets: Tracklets
            object_features: Detected objects features
            matches: Matches between tracklets and detected objects (indices)
        """
        for t_i, d_i in matches:
            tracklet = tracklets[t_i]
            emb = tracklet.get(TrackletCommonData.APPEARANCE)
            if emb is not None:
                emb = object_features[d_i]
            else:
                emb: np.ndarray
                emb = self._appearance_ema_momentum * emb + (1 - self._appearance_ema_momentum) * object_features[d_i]
                emb /= np.linalg.norm(emb)

            tracklet.set(TrackletCommonData.APPEARANCE, emb)
