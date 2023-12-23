"""
Definition of tracker interface.
"""
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

from motrack.library.cv.bbox import PredBBox
from motrack.tracker.tracklet import Tracklet


class Tracker(ABC):
    """
    Tracker interface.
    """
    def __init__(self):
        self._scene: Optional[str] = None  # Optional for scene specific parameters (etc...)

    def set_scene(self, scene: str) -> None:
        """
        Sets scene value

        Args:
            scene:

        Returns:

        """
        self._scene = scene

    def get_scene(self) -> Optional[str]:
        return self._scene

    @abstractmethod
    def track(
        self,
        tracklets: List[Tracklet],
        detections: List[PredBBox],
        frame_index: int,
        frame: Optional[np.ndarray] = None
    ) -> List[Tracklet]:
        """
        Performs multi-object-tracking step.

        Args:
            tracklets: List of active trackelts
            detections: Lists of new detections
            frame_index: Current frame number
            frame: Pass frame in case CMC or appearance based association is used

        Returns:
            Tracks (active, lost, new, deleted, ...)
        """
        pass

    def reset_state(self) -> None:
        """
        Resets tracker state.
        """
        pass
