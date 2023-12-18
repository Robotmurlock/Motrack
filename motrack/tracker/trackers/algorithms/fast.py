"""
Implementation of SparseTrack.
Reference: https://arxiv.org/abs/2306.05238
"""
from typing import Optional, Dict, Any

from motrack.tracker.trackers.algorithms.sort import SortTracker
from motrack.tracker.trackers.catalog import TRACKER_CATALOG


@TRACKER_CATALOG.register('fast')
class FastTracker(SortTracker):
    """
    FastTracker = Just IoU without any filters
    """
    def __init__(
        self,
        filter_name: str = 'no-motion',
        filter_params: Optional[dict] = None,
        cmc_name: Optional[str] = None,
        cmc_params: Optional[dict] = None,
        reid_name: Optional[str] = None,
        reid_params: Optional[str] = None,
        matcher_algorithm: str = 'default',
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

        if matcher_algorithm == 'default':
            matcher_algorithm = 'iou'
            matcher_params = {
                'match_threshold': 0.2,
                'fast_linear_assignment': True
            }

        super().__init__(
            filter_name=filter_name,
            filter_params=filter_params,
            matcher_algorithm=matcher_algorithm,
            matcher_params=matcher_params,
            cmc_name=cmc_name,
            cmc_params=cmc_params,
            reid_name=reid_name,
            reid_params=reid_params,
            remember_threshold=remember_threshold,
            initialization_threshold=initialization_threshold,
            new_tracklet_detection_threshold=new_tracklet_detection_threshold,
            use_observation_if_lost=use_observation_if_lost
        )
