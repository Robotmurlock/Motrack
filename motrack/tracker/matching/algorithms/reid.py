"""
Standard DeepSORT association method based on IoU and appearance info.
"""
from typing import List
from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist

from motrack.library.cv import PredBBox
from motrack.tracker.matching.algorithms.base import AssociationAlgorithm
from motrack.tracker.matching.algorithms.compose import ComposeAssociationAlgorithm
from motrack.tracker.matching.algorithms.iou import LabelGatingType, IoUAssociation
from motrack.tracker.matching.catalog import ASSOCIATION_CATALOG
from motrack.tracker.tracklet import TrackletCommonData, Tracklet


@ASSOCIATION_CATALOG.register('reid')
class ReIDAssociation(AssociationAlgorithm):
    """
    Standard ReID association method appearance.
    """
    def __init__(
        self,
        appearance_threshold: float = 0.0,
        appearance_metric: str = 'cosine',
        fast_linear_assignment: bool = False
    ):
        """
        Args:
            appearance_threshold: Appearance metric minimum threshold
                - Disabled by default
            appearance_metric: Appearance metric (default: cosine)
            fast_linear_assignment: Use faster linear assignment
        """
        super().__init__(
            fast_linear_assignment=fast_linear_assignment,
        )

        self._appearance_threshold = appearance_threshold
        self._appearance_metric: str = appearance_metric

    def form_cost_matrix(
        self,
        tracklet_estimations: List[PredBBox],
        detections: List[PredBBox],
        object_features: Optional[np.ndarray] = None,
        tracklets: Optional[List[Tracklet]] = None
    ) -> np.ndarray:
        assert tracklets is not None, 'Tracklets are mandatory for ReID based association!'
        assert object_features is not None, 'Detection features are also mandatory!'

        n_detections = object_features.shape[0]
        n_tracklets = len(tracklets)

        if n_tracklets == 0 or n_detections == 0:
            return np.zeros(shape=(n_tracklets, n_detections), dtype=np.float32)

        tracklet_features = self._get_features(tracklets)
        # noinspection PyTypeChecker
        cost_matrix = np.maximum(0.0, cdist(tracklet_features, object_features, metric=self._appearance_metric))
        cost_matrix[cost_matrix > 1 - self._appearance_threshold] = np.inf
        return cost_matrix

    @staticmethod
    def _get_features(tracklets: List[Tracklet]) -> np.ndarray:
        """
        Extracts appearance features from the tracklets.

        Args:
            tracklets: List of tracklets

        Returns:
            Tracklet appearance features
        """
        return np.stack([t.get(TrackletCommonData.APPEARANCE) for t in tracklets])


@ASSOCIATION_CATALOG.register('long-term-reid')
class LongTermReIdAssociation(ReIDAssociation):
    """
    Long-term based ReID association.
    """
    def __init__(
        self,
        appearance_threshold: float = 0.0,
        appearance_metric: str = 'cosine',
        fast_linear_assignment: bool = False
    ):
        """
        Args:
            appearance_threshold: Appearance metric minimum threshold
                - Disabled by default
            appearance_metric: Appearance metric (default: cosine)
            fast_linear_assignment: Use faster linear assignment
        """
        super().__init__(
            fast_linear_assignment=fast_linear_assignment,
            appearance_threshold=appearance_threshold,
            appearance_metric=appearance_metric
        )

    @staticmethod
    def _get_features(tracklets: List[Tracklet]) -> np.ndarray:
        features = np.stack([np.stack(list(t.get(TrackletCommonData.APPEARANCE_BUFFER))).mean(axis=0) for t in tracklets])
        return features / np.linalg.norm(features)


@ASSOCIATION_CATALOG.register('reid-iou')
class ReIDIoUAssociation(ComposeAssociationAlgorithm):
    """
    Standard DeepSORT association method based on IoU and appearance.
    """
    def __init__(
        self,
        appearance_weight: float = 0.5,
        appearance_threshold: float = 0.0,
        appearance_metric: str = 'cosine',
        match_threshold: float = 0.30,
        fuse_score: bool = False,
        label_gating: Optional[LabelGatingType] = None,
        fast_linear_assignment: bool = False
    ):
        """
        Args:
            appearance_weight: Appearance to IoU association weight
            appearance_threshold: Appearance metric minimum threshold
            appearance_metric: Appearance metric (default: cosine)
            match_threshold: IoU match threshold
            fuse_score: Fuse detection score with IoU match
            label_gating: Allow different labels to be matched
            fast_linear_assignment: Use faster linear assignment
        """
        assert 0 <= appearance_weight <= 1.0, '"appearance_weight" must be in interval [0, 1]!'

        matchers = [
            IoUAssociation(
                match_threshold=match_threshold,
                fuse_score=fuse_score,
                label_gating=label_gating
            ),
            ReIDAssociation(
                appearance_threshold=appearance_threshold,
                appearance_metric=appearance_metric
            )
        ]

        weights = [1 - appearance_weight, appearance_weight]

        super().__init__(
            matchers=matchers,
            weights=weights,
            fast_linear_assignment=fast_linear_assignment
        )
