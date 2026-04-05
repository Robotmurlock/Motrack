"""
Implementation of Sort tracker with a custom filter.
"""
from typing import Optional, List

import numpy as np
from pydantic import Field

from motrack.library.cv.bbox import PredBBox
from motrack.tracker.matching import association_factory
from motrack.tracker.trackers.algorithms.motion_reid import FactoryConfig, MotionReIDBasedTracker, MotionReIDTrackerConfig
from motrack.tracker.trackers.catalog import TRACKER_CATALOG
from motrack.tracker.tracklet import Tracklet


@TRACKER_CATALOG.register_config('sort')
class SortTrackerConfig(MotionReIDTrackerConfig):
    """
    Config for SORT trackers.
    """

    matcher: FactoryConfig = Field(default_factory=lambda: FactoryConfig(name='iou'))
    duplicate_iou_threshold: float = Field(default=1.0, ge=0.0, le=1.0)
    appearance_ema_momentum: float = Field(default=0.95, ge=0.0, le=1.0)
    appearance_buffer: int = Field(default=0, ge=0)


@TRACKER_CATALOG.register('sort')
class SortTracker(MotionReIDBasedTracker):
    """
    Baseline Sort tracker components:
    - ReID: HungarianIOU
    - Motion model: Filter prior
    - Combining detection and motion model: Filter posterior
    """
    def __init__(self, config: SortTrackerConfig):
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
        super().__init__(
            config=config,
            reid_detection_threshold=None,
            duplicate_iou_threshold=config.duplicate_iou_threshold,
            appearance_ema_momentum=config.appearance_ema_momentum,
            appearance_buffer=config.appearance_buffer
        )

        self._matcher = association_factory(name=config.matcher.name, params=config.matcher.params)

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
