"""
Definition of tracker interface.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple

from motrack.library.cv.bbox import PredBBox
from motrack.tracker.tracklet import Tracklet


class Tracker(ABC):
    """
    Tracker interface.
    """
    @abstractmethod
    def track(
        self,
        tracklets: List[Tracklet],
        detections: List[PredBBox],
        frame_index: int,
        inplace: bool = True
    ) -> Tuple[List[Tracklet], List[Tracklet]]:
        """
        Performs multi-object-tracking step.
        DISCLAIMER: `inplace=True` is default configuration!

        Args:
            tracklets: List of active trackelts
            detections: Lists of new detections
            frame_index: Current frame number
            inplace: Perform inplace transformations on tracklets and bboxes

        Returns:
            - Active tracklets
            - All tracklets
        """
        pass
