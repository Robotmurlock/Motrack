"""
Camera motion compensation interface.
"""
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class CameraMotionCompensation(ABC):
    @abstractmethod
    def apply(self, frame: np.ndarray, frame_index: int, scene: Optional[str] = None) -> np.ndarray:
        """
        Calculates approximated affine transformations applied
        to the image between current and last frame.

        Args:
            frame: Current video frame
            frame_index: Frame index
            scene: Scene name (optional)

        Returns:
            Affine 2x3 matrix (includes translation)
        """
