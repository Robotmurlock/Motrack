"""
Implementation of ByteTrack.
Reference: https://arxiv.org/abs/2110.06864
"""
from typing import Optional, Dict, Any, List

import numpy as np

from motrack.library.cv.bbox import PredBBox
from motrack.tracker.matching import association_factory
from motrack.tracker.trackers.algorithms.motion_reid import MotionReIDBasedTracker
from motrack.tracker.trackers.catalog import TRACKER_CATALOG
from motrack.tracker.trackers.utils import remove_duplicates
from motrack.tracker.tracklet import Tracklet, TrackletState
from motrack.utils.collections import unpack_n


@TRACKER_CATALOG.register('byte')
class ByteTracker(MotionReIDBasedTracker):
    """
    ByteTrack algorithm.

    Steps:
        0. Estimate all tracklets priors
        1. Split detections into high and low
        2. Match high detections with tracklets with states ACTIVE and LOST using HighMatchAlgorithm
        3. Match remaining ACTIVE tracklets with low detections using LowMatchAlgorithm
        4. Mark remaining (unmatched) tracklets as lost (performed with step 8)
        5. Match NEW tracklets with high detections using NewMatchAlgorithm
            - remove all NEW unmatched tracklets
        6. Initialize new tracklets from unmatched high detections
        7. Update matched tracklets
        8. Delete new unmatched and long-lost tracklets
        9. Delete duplicate between ACTIVE and LOST tracklets
    """
    def __init__(
        self,
        filter_name: str = 'bot-sort',
        filter_params: Optional[dict] = None,
        cmc_name: Optional[str] = None,
        cmc_params: Optional[dict] = None,
        reid_name: Optional[str] = None,
        reid_params: Optional[str] = None,
        high_matcher_algorithm: str = 'default',
        high_matcher_params: Optional[Dict[str, Any]] = None,
        low_matcher_algorithm: str = 'default',
        low_matcher_params: Optional[Dict[str, Any]] = None,
        new_matcher_algorithm: str = 'default',
        new_matcher_params: Optional[Dict[str, Any]] = None,
        detection_threshold: float = 0.6,
        remember_threshold: int = 1,
        initialization_threshold: int = 3,
        new_tracklet_detection_threshold: Optional[float] = None,
        duplicate_iou_threshold: float = 0.85,
        use_observation_if_lost: bool = False,
        appearance_ema_momentum: float = 0.95
    ):
        if filter_params is None:
            filter_params = {}

        new_tracklet_detection_threshold = new_tracklet_detection_threshold if new_tracklet_detection_threshold is not None \
            else detection_threshold
        super().__init__(
            filter_name=filter_name,
            filter_params=filter_params,
            cmc_name=cmc_name,
            cmc_params=cmc_params,
            reid_name=reid_name,
            reid_params=reid_params,

            new_tracklet_detection_threshold=new_tracklet_detection_threshold,
            remember_threshold=remember_threshold,
            initialization_threshold=initialization_threshold,
            use_observation_if_lost=use_observation_if_lost,

            appearance_ema_momentum=appearance_ema_momentum,
            reid_detection_threshold=detection_threshold
        )

        if high_matcher_algorithm == 'default':
            assert high_matcher_params is None
            high_matcher_algorithm = 'iou'
            high_matcher_params = {
                'match_threshold': 0.2,
                'fuse_score': True
            }

        self._high_match = association_factory(high_matcher_algorithm, high_matcher_params)

        if low_matcher_algorithm == 'default':
            assert low_matcher_params is None
            low_matcher_algorithm = 'iou'
            low_matcher_params = {
                'match_threshold': 0.5
            }

        self._low_match = association_factory(low_matcher_algorithm, low_matcher_params)

        if new_matcher_algorithm == 'default':
            assert new_matcher_params is None
            new_matcher_algorithm = 'iou'
            new_matcher_params = {
                'match_threshold': 0.3,
                'fuse_score': True
            }

        self._new_match = association_factory(new_matcher_algorithm, new_matcher_params)

        # Parameters
        self._initialization_threshold = initialization_threshold
        self._remember_threshold = remember_threshold
        self._use_observation_if_lost = use_observation_if_lost
        self._detection_threshold = detection_threshold
        self._duplicate_iou_threshold = duplicate_iou_threshold

        # State
        self._next_id = 0

    def _track(
        self,
        tracklets: List[Tracklet],
        prior_tracklet_bboxes: List[PredBBox],
        detections: List[PredBBox],
        frame_index: int,
        objects_features: Optional[np.ndarray] = None,
        frame: Optional[np.ndarray] = None
    ) -> List[Tracklet]:
        # (1) Split detections into low and high
        high_detections = [d for d in detections if d.conf >= self._detection_threshold]
        high_det_indices = [i for i, d in enumerate(detections) if d.conf >= self._detection_threshold]
        low_detections = [d for d in detections if d.conf < self._detection_threshold]
        low_det_indices = [i for i, d in enumerate(detections) if d.conf < self._detection_threshold]

        # (2) Match high detections with tracklets with states ACTIVE and LOST using HighMatchAlgorithm
        tracklets_active_and_lost_indices, tracklets_active_and_lost, tracklets_active_and_lost_bboxes = \
            unpack_n([(i, t, t_bbox) for i, (t, t_bbox) in enumerate(zip(tracklets, prior_tracklet_bboxes)) if t.is_tracked], n=3)
        high_matches, remaining_tracklet_indices, high_unmatched_detections_indices = \
            self._high_match(tracklets_active_and_lost_bboxes, high_detections,
                             object_features=objects_features, tracklets=tracklets_active_and_lost)
        high_matches = [(tracklets_active_and_lost_indices[t_i], high_det_indices[d_i]) for t_i, d_i in high_matches]
        high_unmatched_detections_indices = [high_det_indices[d_i] for d_i in high_unmatched_detections_indices]
        remaining_tracklet_bboxes = [tracklets_active_and_lost_bboxes[t_i] for t_i in remaining_tracklet_indices]
        remaining_tracklets = [tracklets_active_and_lost[t_i] for t_i in remaining_tracklet_indices]
        remaining_tracklet_indices = [tracklets_active_and_lost_indices[t_i] for t_i in remaining_tracklet_indices]

        # (3) Match remaining ACTIVE tracklets with low detections using LowMatchAlgorithm
        remaining_active_tracklet_indices, remaining_active_tracklets, remaining_active_tracklet_bboxes = \
            unpack_n([(i, t, t_bbox) for i, t, t_bbox in zip(remaining_tracklet_indices, remaining_tracklets, remaining_tracklet_bboxes)
                  if t.state == TrackletState.ACTIVE], n=3)
        low_matches, low_unmatched_tracklet_indices, _ = \
            self._low_match(remaining_active_tracklet_bboxes, low_detections,
                            object_features=None, tracklets=remaining_active_tracklets)
        low_matches = [(remaining_active_tracklet_indices[t_i], low_det_indices[d_i]) for t_i, d_i in low_matches]
        unmatched_tracklet_indices = [remaining_active_tracklet_indices[t_i] for t_i in low_unmatched_tracklet_indices]

        # (5) Match NEW tracklets with high detections using NewMatchAlgorithm
        remaining_high_detections = [detections[d_i] for d_i in high_unmatched_detections_indices]
        remaining_high_detection_indices = high_unmatched_detections_indices
        tracklets_new_indices, tracklets_new, tracklets_new_bboxes = \
            unpack_n([(i, t, t_bbox) for i, (t, t_bbox) in enumerate(zip(tracklets, prior_tracklet_bboxes)) if t.state == TrackletState.NEW], n=3)
        new_matches, new_unmatched_tracklets_indices, new_unmatched_detections_indices = \
            self._new_match(tracklets_new_bboxes, remaining_high_detections,
                            object_features=None, tracklets=tracklets_new)
        new_matches = [(tracklets_new_indices[t_i], high_unmatched_detections_indices[d_i]) for t_i, d_i in new_matches]
        new_unmatched_tracklets_indices = [tracklets_new_indices[t_i] for t_i in new_unmatched_tracklets_indices]
        new_unmatched_detections_indices = [remaining_high_detection_indices[d_i] for d_i in new_unmatched_detections_indices]

        # (6) Initialize new tracklets from unmatched high detections
        new_tracklets = self._create_new_tracklets(
            detections=[detections[d_i] for d_i in new_unmatched_detections_indices],
            frame_index=frame_index,
            objects_features=objects_features[new_unmatched_detections_indices] if objects_features is not None else None
        )

        # (7) Update matched tracklets
        all_matches = high_matches + low_matches + new_matches
        matched_detection_indices = [d_i for _, d_i in all_matches]
        matched_objects_features: Optional[List[Optional[np.ndarray]]] = None
        if objects_features is not None:
            # Object's that have features (because of the high detection score)
            # have stored features in the list, otherwise None is set
            detection_to_object_feature_index = {d_i: i for i, d_i in enumerate(high_det_indices)}
            matched_features_indices = [detection_to_object_feature_index.get(d_i) for d_i in matched_detection_indices]
            matched_objects_features = [(objects_features[i] if i is not None else None) for i in matched_features_indices]

        self._update_tracklets(
            matched_tracklets=[tracklets[t_i] for t_i, _ in all_matches],
            matched_detections=[detections[d_i] for d_i in matched_detection_indices],
            matched_object_features=matched_objects_features,
            frame_index=frame_index
        )

        # (8) Delete new unmatched and long-lost tracklets
        self._handle_lost_tracklets(
            lost_tracklets=[tracklets[t_i] for t_i in unmatched_tracklet_indices + new_unmatched_tracklets_indices]
        )

        # (9) Delete duplicate between ACTIVE and LOST tracklets
        deleted_tracklets = [t for t in tracklets if t.state == TrackletState.DELETED]
        active_tracklets = [t for t in tracklets if t.state == TrackletState.ACTIVE]
        lost_tracklets = [t for t in tracklets if t.state == TrackletState.LOST]
        new_tracklets.extend([t for t in tracklets if t.state == TrackletState.NEW])  # Add new tracklets from previous iteration
        active_tracklets, lost_tracklets = remove_duplicates(self._duplicate_iou_threshold, active_tracklets, lost_tracklets)
        return active_tracklets + lost_tracklets + new_tracklets + deleted_tracklets
