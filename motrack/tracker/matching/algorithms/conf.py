"""
Implementation of Confidence association inspired by the Hybrid-SORT paper.
Reference: https://arxiv.org/pdf/2308.00783.pdf
"""
from typing import List, Optional

import numpy as np

from motrack.library.cv import PredBBox
from motrack.tracker.matching.algorithms.base import AssociationAlgorithm
from motrack.tracker.tracklet import Tracklet, TrackletState
from motrack.library.kalman_filter.conf_kf import ConfidenceKalmanFilter
from scipy.spatial.distance import cdist
from motrack.tracker.matching.catalog import ASSOCIATION_CATALOG


@ASSOCIATION_CATALOG.register('hybrid-conf')
class HybridConfidenceAssociation(AssociationAlgorithm):
    """
    Hybrid-SORT inspired confidence based association algorithm.
    """
    DATA_KEY = 'kf-conf-est'

    def __init__(
        self,
        initial_P_conf: float = 10.0,
        Q_conf: float = 1.0,
        Q_conf_velocity: float = 1e-3,
        R_conf: float = 100.0,
        linear_prediction: bool = False,
        fast_linear_assignment: bool = False
    ):
        """
        Args:
            initial_P_conf: Initial confidence uncertainty (std)
            Q_conf: Initial process noise confidence "position"
            Q_conf_velocity: Initial process noise confidence "velocity"
            R_conf: Confidence noise uncertainty
            linear_prediction: Use linear prediction instead of KF
            fast_linear_assignment: Use greedy algorithm for linear assignment
                - This might be more efficient in case of large cost matrix
        """
        super().__init__(fast_linear_assignment=fast_linear_assignment)
        self._kf = ConfidenceKalmanFilter(
            initial_P_conf=initial_P_conf,
            Q_conf=Q_conf,
            Q_conf_velocity=Q_conf_velocity,
            R_conf=R_conf
        )
        self._linear_prediction = linear_prediction

    def _predict_conf(self, tracklet: Tracklet) -> float:
        """
        Predicts confidence for next step.
        Uses linear (naive) extrapolation or KF.

        Args:
            tracklet: Tracklet

        Returns:
            Predicted confidence
        """
        if self._linear_prediction:
            # Perform linear prediction
            history_len = len(tracklet.history)
            if history_len < 2:
                return tracklet.bbox.conf

            last_bbox_frame_index, last_bbox = tracklet.history[-1]
            last_bbox_conf = last_bbox.conf
            prev_bbox_frame_index, prev_bbox = tracklet.history[-2]
            prev_bbox_conf = prev_bbox.conf

            delta = (last_bbox_conf - prev_bbox_conf) / (last_bbox_frame_index - prev_bbox_frame_index)
            conf = last_bbox_conf + delta * (1 + tracklet.lost_time)
        else:
            # Perform KF prediction
            mean, std = tracklet.get(self.DATA_KEY)
            mean, std = self._kf.predict(mean, std)
            tracklet.set(self.DATA_KEY, (mean, std))

            conf = float(self._kf.project(mean, std)[0])

        return max(0.0, min(1.0, conf))

    def _update_conf(self, tracklets: List[Tracklet]) -> None:
        """
        Updates KF state with new confidence "measurements".

        Args:
            tracklets: List of tracklets
        """
        for tracklet in tracklets:
            if tracklet.state == TrackletState.LOST:
                continue

            conf_measurement = np.array([tracklet.bbox.conf], dtype=np.float32)

            state = tracklet.get(self.DATA_KEY)
            if state is None:
                mean, std = self._kf.initiate(conf_measurement)
            else:
                mean, std = state
                mean, std = self._kf.update(mean, std, conf_measurement)

            tracklet.set(self.DATA_KEY, (mean, std))

    def form_cost_matrix(
        self,
        tracklet_estimations: List[PredBBox],
        detections: List[PredBBox],
        object_features: Optional[np.ndarray] = None,
        tracklets: Optional[List[Tracklet]] = None
    ) -> np.ndarray:
        _, _ = tracklet_estimations, object_features  # Unused

        self._update_conf(tracklets)
        tracklet_conf_estimations = np.array([self._predict_conf(t) for t in tracklets], dtype=np.float32).reshape(-1, 1)
        detection_confs = np.array([bbox.conf for bbox in detections], dtype=np.float32).reshape(-1, 1)
        return np.maximum(0.0, cdist(tracklet_conf_estimations, detection_confs, metric='cityblock'))
