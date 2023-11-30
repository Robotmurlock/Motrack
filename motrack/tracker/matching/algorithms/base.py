"""
Association method interface.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from motrack.library.cv.bbox import PredBBox
from motrack.tracker.tracklet import Tracklet


class AssociationAlgorithm(ABC):
    """
    Defines interface to tracklets-detections association matching.
    """
    @abstractmethod
    def match(
        self,
        tracklet_estimations: List[PredBBox],
        detections: List[PredBBox],
        tracklets: Optional[List[Tracklet]] = None
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Performs matching between tracklets and detections (interface).

        Args:
            tracklet_estimations: Tracked object from previous frames
            detections: Currently detected objects
            tracklets: Full tracklet info (optional)

        Returns:
            - List of matches (pairs)
            - list of unmatched tracklets
            - list of unmatched detections
        """
        pass

    def __call__(
        self,
        tracklet_estimations: List[PredBBox],
        detections: List[PredBBox],
        tracklets: Optional[List[Tracklet]] = None
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        return self.match(tracklet_estimations, detections, tracklets=tracklets)
