"""
Implementation of Sort tracker with a custom filter.
"""
import copy
from typing import Optional, Dict, Any, List

import numpy as np

from motrack.library.cv.bbox import PredBBox
from motrack.tracker.matching import association_factory
from motrack.tracker.trackers.algorithms.motion_reid import MotionReIDBasedTracker
from motrack.tracker.trackers.catalog import TRACKER_CATALOG
from motrack.tracker.tracklet import Tracklet, TrackletState


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
        use_observation_if_lost: bool = False
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

            new_tracklet_detection_threshold=new_tracklet_detection_threshold
        )

        matcher_params = {} if matcher_params is None else matcher_params
        self._matcher = association_factory(name=matcher_algorithm, params=matcher_params)

        # Parameters
        self._remember_threshold = remember_threshold
        self._initialization_threshold = initialization_threshold
        self._use_observation_if_lost = use_observation_if_lost

    def _track(
        self,
        tracklets: List[Tracklet],
        prior_tracklet_bboxes: List[PredBBox],
        detections: List[PredBBox],
        frame_index: int,
        objects_features: Optional[np.ndarray] = None,
        inplace: bool = True,
        frame: Optional[np.ndarray] = None
    ) -> List[Tracklet]:
        # Perform matching
        matches, unmatched_tracklets, unmatched_detections = self._matcher(prior_tracklet_bboxes, detections,
                                                                           object_features=objects_features, tracklets=tracklets)

        # Update matched tracklets
        for tracklet_index, det_index in matches:
            tracklet = tracklets[tracklet_index] if inplace else copy.deepcopy(tracklets[tracklet_index])
            det_bbox = detections[det_index] if inplace else copy.deepcopy(detections[det_index])
            tracklet_bbox, _, _ = self._update(tracklets[tracklet_index], det_bbox)
            new_bbox = det_bbox if self._use_observation_if_lost and tracklet.state != TrackletState.ACTIVE \
                else tracklet_bbox

            new_state = TrackletState.ACTIVE
            if tracklet.state == TrackletState.NEW and tracklet.total_matches + 1 < self._initialization_threshold:
                new_state = TrackletState.NEW
            tracklets[tracklet_index] = tracklet.update(new_bbox, frame_index, state=new_state)

        # Create new tracklets from unmatched detections and initiate filter states
        new_tracklets = self._create_new_tracklets(
            detections=[detections[d_i] for d_i in unmatched_detections],
            frame_index=frame_index,
            objects_features=objects_features[unmatched_detections] if objects_features is not None else None,
            inplace=inplace
        )

        # Delete old and new unmatched tracks
        for tracklet_index in unmatched_tracklets:
            tracklet = tracklets[tracklet_index]
            if tracklet.number_of_unmatched_frames(frame_index) > self._remember_threshold \
                    or tracklet.state == TrackletState.NEW:
                self._delete(tracklet.id)
                tracklet.state = TrackletState.DELETED
            else:
                tracklet_bbox, _, _ = self._missing(tracklet)
                tracklets[tracklet_index] = tracklet.update(tracklet_bbox, tracklet.frame_index, state=TrackletState.LOST)

        return tracklets + new_tracklets
