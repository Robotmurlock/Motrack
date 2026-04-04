"""
Implementation of SparseTrack.
Reference: https://arxiv.org/abs/2306.05238
"""
from pydantic import Field

from motrack.tracker.trackers.algorithms.byte import ByteTracker, ByteTrackerConfig
from motrack.tracker.trackers.algorithms.motion_reid import FactoryConfig
from motrack.tracker.trackers.catalog import TRACKER_CATALOG


@TRACKER_CATALOG.register_config('sparse')
class SparseTrackerConfig(ByteTrackerConfig):
    """
    Config for SparseTrack trackers.
    """

    high_matcher: FactoryConfig = Field(
        default_factory=lambda: FactoryConfig(name='dcm', params={'levels': 12, 'match_threshold': 0.2, 'fuse_score': True})
    )
    low_matcher: FactoryConfig = Field(
        default_factory=lambda: FactoryConfig(name='dcm', params={'levels': 12, 'match_threshold': 0.5})
    )
    new_matcher: FactoryConfig = Field(
        default_factory=lambda: FactoryConfig(name='iou', params={'match_threshold': 0.3, 'fuse_score': True})
    )
    duplicate_iou_threshold: float = Field(default=0.85, ge=0.0, le=1.0)


@TRACKER_CATALOG.register('sparse')
class SparseTracker(ByteTracker):
    """
    SparseTrack = ByteTrack + DCM
    """
    def __init__(self, config: SparseTrackerConfig):
        super().__init__(config)
