"""
Implementation of SparseTrack.
"""
from typing import Optional, Dict, Any

from nodetracker.tracker.trackers.byte import ByteTracker


class SparseTracker(ByteTracker):
    """
    SparseTrack = ByteTrack + DCM
    """
    def __init__(
        self,
        filter_name: str,
        filter_params: dict,
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
        use_observation_if_lost: bool = False
    ):

        if high_matcher_algorithm == 'default':
            assert high_matcher_params is None
            high_matcher_algorithm = 'dcm'
            high_matcher_params = {
                'levels': 12,  # Default for DanceTrack
                'match_threshold': 0.2,
                'fuse_score': True
            }

        if low_matcher_algorithm == 'default':
            assert low_matcher_params is None
            low_matcher_algorithm = 'dcm'
            low_matcher_params = {
                'levels': 12,  # Default for DanceTrack
                'match_threshold': 0.5
            }

        if new_matcher_algorithm == 'default':
            assert new_matcher_params is None
            new_matcher_algorithm = 'hungarian_iou'
            new_matcher_params = {
                'match_threshold': 0.3,
                'fuse_score': True
            }

        super().__init__(
            filter_name=filter_name,
            filter_params=filter_params,
            high_matcher_algorithm=high_matcher_algorithm,
            high_matcher_params=high_matcher_params,
            low_matcher_algorithm=low_matcher_algorithm,
            low_matcher_params=low_matcher_params,
            new_matcher_algorithm=new_matcher_algorithm,
            new_matcher_params=new_matcher_params,
            detection_threshold=detection_threshold,
            remember_threshold=remember_threshold,
            initialization_threshold=initialization_threshold,
            new_tracklet_detection_threshold=new_tracklet_detection_threshold,
            duplicate_iou_threshold=duplicate_iou_threshold,
            use_observation_if_lost=use_observation_if_lost
        )
