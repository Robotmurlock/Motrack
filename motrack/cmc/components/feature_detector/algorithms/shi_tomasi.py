"""
Shi-Tomasi corner detector.
"""
from typing import ClassVar, Optional

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from motrack.cmc.components.feature_detector.algorithms.base import DescriptorNorm, FeatureDetector
from motrack.cmc.components.feature_detector.utils import empty_points
from motrack.cmc.components.feature_detector.catalog import FEATURE_DETECTOR_CATALOG


@FEATURE_DETECTOR_CATALOG.register_config('shi-tomasi')
class ShiTomasiFeatureDetectorConfig(BaseModel):
    """
    Config for the Shi-Tomasi corner detector.

    Attributes:
        max_features: How many corners to keep, strongest first. Each one is tracked
            separately, so this drives the runtime.
        quality_level: Minimum corner strength, as a fraction of the strongest corner in the
            image. Being relative, it adapts to how bright or contrasty the frame is.
        min_distance: Minimum spacing between corners, in pixels. Stops them bunching up in
            one textured area and leaving the rest of the frame unconstrained.
        block_size: Size of the window used to score each corner. Larger is more stable but
            less precise.
        use_harris_detector: Use the Harris corner score instead of the minimum eigenvalue.
            Lucas-Kanade uses the minimum eigenvalue too, so leaving this off keeps detection
            and tracking consistent.
        k: Harris tuning constant. Only used when `use_harris_detector` is set.
    """

    model_config = ConfigDict(extra='forbid')

    max_features: int = Field(default=1000, gt=0)
    quality_level: float = Field(default=0.01, gt=0.0, le=1.0)
    min_distance: float = Field(default=7.0, gt=0.0)
    block_size: int = Field(default=3, ge=1)
    use_harris_detector: bool = False
    k: float = Field(default=0.04, gt=0.0)


@FEATURE_DETECTOR_CATALOG.register('shi-tomasi')
class ShiTomasiFeatureDetector(FeatureDetector):
    """
    Shi-Tomasi corner detector.

    Ranks corners by the minimum eigenvalue of the local structure matrix, which is the same
    quantity the Lucas-Kanade tracker uses to decide whether a point is trackable at all.
    Produces no descriptors, so it can only be paired with tracking based correspondence.
    """
    produces_descriptors: ClassVar[bool] = False
    descriptor_norm: ClassVar[Optional[DescriptorNorm]] = None

    def __init__(self, config: ShiTomasiFeatureDetectorConfig):
        """
        Args:
            config: Shi-Tomasi config
        """
        self._config = config

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray]]:
        corners = cv2.goodFeaturesToTrack(
            frame,
            maxCorners=self._config.max_features,
            qualityLevel=self._config.quality_level,
            minDistance=self._config.min_distance,
            blockSize=self._config.block_size,
            useHarrisDetector=self._config.use_harris_detector,
            k=self._config.k
        )

        if corners is None:
            return empty_points(), None

        return corners.reshape(-1, 2).astype(np.float32), None
