"""
Implementation of Sort tracker with a custom filter.
"""
from typing import Optional, Dict, Any, List

import numpy as np

from motrack.library.cv.bbox import PredBBox
from motrack.tracker.matching import association_factory
from motrack.tracker.trackers.algorithms.motion_reid import MotionReIDBasedTracker
from motrack.tracker.trackers.catalog import TRACKER_CATALOG
from motrack.tracker.tracklet import Tracklet


@TRACKER_CATALOG.register('sort')
class SortTracker(MotionReIDBasedTracker):
    """
    Baseline Sort tracker components:
    - ReID: HungarianIOU
    - Motion model: Filter prior
    - Combining detection and motion model: Filter posterior
    """
    def __init__(
        self,
        filter_name: str = 'bot-sort',
        filter_params: Optional[dict] = None,
        cmc_name: Optional[str] = None,
        cmc_params: Optional[dict] = None,
        reid_name: Optional[str] = None,
        reid_params: Optional[str] = None,
        matcher_algorithm: str = 'iou',
        matcher_params: Optional[Dict[str, Any]] = None,
        remember_threshold: int = 1,
        initialization_threshold: int = 3,
        new_tracklet_detection_threshold: Optional[float] = None,
        use_observation_if_lost: bool = False,
        duplicate_iou_threshold: float = 1.00,
        appearance_ema_momentum: float = 0.95,
        appearance_buffer: int = 0
    ):
        """
        Args:
            filter_name: Filter name
            filter_params: Filter params
            cmc_name: CMC name
            cmc_params: CMC params
            reid_name: ReID name
            reid_params: ReID params

            matcher_algorithm: Choose matching algorithm (e.g. Hungarian IOU)
            matcher_params: Matching algorithm parameters

            remember_threshold: How long does the tracklet without any detection matching
                If tracklet isn't matched for more than `remember_threshold` frames then
                it is deleted.
            initialization_threshold: Number of frames until tracklet becomes active
            new_tracklet_detection_threshold: Threshold to accept new tracklet
            use_observation_if_lost: When re-finding tracklet, use observation instead of estimation
            appearance_buffer: Appearance buffer length (set to a non-zero non-negative integer to activate)
        """
        if filter_params is None:
            filter_params = {}

        super().__init__(
            filter_name=filter_name,
            filter_params=filter_params,
            cmc_name=cmc_name,
            cmc_params=cmc_params,
            reid_name=reid_name,
            reid_params=reid_params,
            reid_detection_threshold=None,

            new_tracklet_detection_threshold=new_tracklet_detection_threshold,
            remember_threshold=remember_threshold,
            initialization_threshold=initialization_threshold,
            use_observation_if_lost=use_observation_if_lost,
            duplicate_iou_threshold=duplicate_iou_threshold,

            appearance_ema_momentum=appearance_ema_momentum,
            appearance_buffer=appearance_buffer
        )

        matcher_params = {} if matcher_params is None else matcher_params
        self._matcher = association_factory(name=matcher_algorithm, params=matcher_params)

    def _track(
        self,
        tracklets: List[Tracklet],
        prior_tracklet_bboxes: List[PredBBox],
        detections: List[PredBBox],
        frame_index: int,
        objects_features: Optional[np.ndarray] = None,
        frame: Optional[np.ndarray] = None
    ) -> List[Tracklet]:
        # Perform matching
        matches, unmatched_tracklets, unmatched_detections = self._matcher(prior_tracklet_bboxes, detections,
                                                                           object_features=objects_features, tracklets=tracklets)

        self._update_tracklets(
            matched_tracklets=[tracklets[t_i] for t_i, _ in matches],
            matched_detections=[detections[d_i] for _, d_i in matches],
            matched_object_features=objects_features[[d_i for _, d_i in matches]] if objects_features is not None else None,
            frame_index=frame_index
        )

        # Create new tracklets from unmatched detections and initiate filter states
        new_tracklets = self._create_new_tracklets(
            detections=[detections[d_i] for d_i in unmatched_detections],
            frame_index=frame_index,
            objects_features=objects_features[unmatched_detections] if objects_features is not None else None
        )

        # Delete old and new unmatched tracks
        self._handle_lost_tracklets(
            lost_tracklets=[tracklets[t_i] for t_i in unmatched_tracklets]
        )

        tracklets.extend(new_tracklets)
        return self._postprocess_tracklets(tracklets)
