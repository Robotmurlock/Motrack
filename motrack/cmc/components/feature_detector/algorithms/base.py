"""
Feature detector interface.
"""
from abc import ABC, abstractmethod
from typing import ClassVar, Optional

import numpy as np

# A detector declares which norm its descriptors are compared under, and the distance catalog
# is what that string selects, so the norms live there rather than being restated here.
from motrack.cmc.components.distances import DescriptorNorm


class FeatureDetector(ABC):
    """
    Abstract base class for feature detectors.

    Attributes:
        produces_descriptors: Whether `detect` returns descriptors alongside the points.
            Some detectors, like Shi-Tomasi, do not produce descriptors - just corners as points.
        descriptor_norm: Norm the descriptors are compared under, None when there are none.
    """
    produces_descriptors: ClassVar[bool] = False
    descriptor_norm: ClassVar[Optional[DescriptorNorm]] = None

    @abstractmethod
    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Detect features in the given frame.

        Detection is unconstrained: excluding regions that belong to dynamic objects is done
        by filtering the returned points against the detection boxes, which also covers the
        points a correspondence lands on in the next frame.

        Args:
            frame: Grayscale frame to detect features in.

        Returns:
            Points of shape (N, 2) in (x, y) pixel coordinates, and their descriptors of
            shape (N, D) or None when the detector does not produce any.
        """
        pass
