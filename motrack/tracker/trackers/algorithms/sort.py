"""
Implementation of Sort tracker with a custom filter.
"""
import copy
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import torch

from nodetracker.library.cv.bbox import PredBBox
from nodetracker.tracker.matching import association_algorithm_factory
from nodetracker.tracker.trackers.motion import MotionBasedTracker
from nodetracker.tracker.tracklet import Tracklet, TrackletState


class SortTracker(MotionBasedTracker):
    """
    Baseline Sort tracker components:
    - ReID: HungarianIOU
    - Motion model: Filter prior
    - Combining detection and motion model: Filter posterior
    """
    def __init__(
        self,
        filter_name: str,
        filter_params: dict,
        matcher_algorithm: str = 'hungarian_iou',
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

            matcher_algorithm: Choose matching algorithm (e.g. Hungarian IOU)
            matcher_params: Matching algorithm parameters

            remember_threshold: How long does the tracklet without any detection matching
                If tracklet isn't matched for more than `remember_threshold` frames then
                it is deleted.
            initialization_threshold: Number of frames until tracklet becomes active
            new_tracklet_detection_threshold: Threshold to accept new tracklet
            use_observation_if_lost: When re-finding tracklet, use observation instead of estimation
        """
        super().__init__(
            filter_name=filter_name,
            filter_params=filter_params,
        )

        matcher_params = {} if matcher_params is None else matcher_params
        self._matcher = association_algorithm_factory(name=matcher_algorithm, params=matcher_params)

        # Parameters
        self._remember_threshold = remember_threshold
        self._initialization_threshold = initialization_threshold
        self._new_tracklet_detection_threshold = new_tracklet_detection_threshold
        self._use_observation_if_lost = use_observation_if_lost

    def track(self, tracklets: List[Tracklet], detections: List[PredBBox], frame_index: int, inplace: bool = True) \
            -> Tuple[List[Tracklet], List[Tracklet]]:
        # Estimate priors for all tracklets
        prior_tracklet_estimates = [self._predict(t) for t in tracklets]
        prior_tracklet_bboxes = [bbox for bbox, _, _ in prior_tracklet_estimates]

        # Perform matching
        matches, unmatched_tracklets, unmatched_detections = self._matcher(prior_tracklet_bboxes, detections, tracklets=tracklets)

        # Update matched tracklets
        for tracklet_index, det_index in matches:
            tracklet = tracklets[tracklet_index] if inplace else copy.deepcopy(tracklets[tracklet_index])
            det_bbox = detections[det_index] if inplace else copy.deepcopy(detections[det_index])
            tracklet_bbox, _, _ = self._update(tracklets[tracklet_index], det_bbox)
            new_bbox = det_bbox if self._use_observation_if_lost and tracklet.state != TrackletState.ACTIVE \
                else tracklet_bbox
            tracklets[tracklet_index] = tracklet.update(new_bbox, frame_index, state=TrackletState.ACTIVE)

        # Create new tracklets from unmatched detections and initiate filter states
        new_tracklets: List[Tracklet] = []
        for det_index in unmatched_detections:
            detection = detections[det_index]
            if self._new_tracklet_detection_threshold is not None and detection.conf < self._new_tracklet_detection_threshold:
                continue

            new_tracklet = Tracklet(
                bbox=detection if inplace else copy.deepcopy(detection),
                frame_index=frame_index
            )
            new_tracklets.append(new_tracklet)
            self._initiate(new_tracklet.id, detection)

        # Delete old unmatched tracks
        tracklets_indices_to_delete: List[int] = []
        for tracklet_index in unmatched_tracklets:
            tracklet = tracklets[tracklet_index]
            if tracklet.number_of_unmatched_frames(frame_index) > self._remember_threshold \
                    or tracklet.state == TrackletState.NEW:
                tracklets_indices_to_delete.append(tracklet_index)
                self._delete(tracklet.id)
            else:
                tracklet_bbox, _, _ = self._missing(tracklet)
                tracklets[tracklet_index] = tracklet.update(tracklet_bbox, tracklet.frame_index, state=TrackletState.LOST)

        all_tracklets = [t for i, t in enumerate(tracklets) if i not in tracklets_indices_to_delete] \
            + new_tracklets

        # Filter active tracklets
        active_tracklets = [t for t in tracklets if t.total_matches >= self._initialization_threshold]
        active_tracklets = [t for t in active_tracklets if t.state == TrackletState.ACTIVE]

        return active_tracklets, all_tracklets
