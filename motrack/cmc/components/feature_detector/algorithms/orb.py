"""
ORB feature detector.
"""
from typing import ClassVar, Optional

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from motrack.cmc.components.feature_detector.algorithms.base import DescriptorNorm, FeatureDetector
from motrack.cmc.components.feature_detector.utils import pack_keypoints
from motrack.cmc.components.feature_detector.catalog import FEATURE_DETECTOR_CATALOG


@FEATURE_DETECTOR_CATALOG.register_config('orb')
class OrbFeatureDetectorConfig(BaseModel):
    """
    Config for the ORB detector.

    Attributes:
        max_features: How many keypoints to keep. Matching cost grows quadratically with the number of keypoints.
        scale_factor: How much the image shrinks at each pyramid level. Values closer to 1
            handle scale changes better but need more levels.
        n_levels: How many pyramid levels. With `scale_factor`, sets the largest size change
            that can still be matched: `scale_factor ** (n_levels - 1)`.
        edge_threshold: Border in pixels where no keypoints are detected, because the
            descriptor patch would run off the image.
        fast_threshold: How much a pixel must differ from its neighbours to count as a
            corner. Lower finds more corners, including weak and noisy ones.
    """

    model_config = ConfigDict(extra='forbid')

    max_features: int = Field(default=1000, gt=0)
    scale_factor: float = Field(default=1.2, gt=1.0)
    n_levels: int = Field(default=8, ge=1)
    edge_threshold: int = Field(default=31, ge=1)
    fast_threshold: int = Field(default=20, ge=1)


@FEATURE_DETECTOR_CATALOG.register('orb')
class OrbFeatureDetector(FeatureDetector):
    """
    ORB detector and descriptor.

    Produces a 256 bit binary descriptor packed into 32 bytes, compared under the Hamming
    norm. Considerably cheaper than SIFT, which is the reason it is the usual choice for
    camera motion estimation.
    """
    produces_descriptors: ClassVar[bool] = True
    descriptor_norm: ClassVar[Optional[DescriptorNorm]] = 'hamming'

    def __init__(self, config: OrbFeatureDetectorConfig):
        """
        Args:
            config: ORB config
        """
        self._orb = cv2.ORB_create(
            nfeatures=config.max_features,
            scaleFactor=config.scale_factor,
            nlevels=config.n_levels,
            edgeThreshold=config.edge_threshold,
            fastThreshold=config.fast_threshold
        )

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray]]:
        keypoints, descriptors = self._orb.detectAndCompute(frame, None)
        return pack_keypoints(keypoints, descriptors)
