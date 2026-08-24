"""
SIFT feature detector.
"""
from typing import ClassVar, Optional

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from motrack.cmc.components.feature_detector.algorithms.base import DescriptorNorm, FeatureDetector
from motrack.cmc.components.feature_detector.utils import pack_keypoints
from motrack.cmc.components.feature_detector.catalog import FEATURE_DETECTOR_CATALOG


@FEATURE_DETECTOR_CATALOG.register_config('sift')
class SiftFeatureDetectorConfig(BaseModel):
    """
    Config for the SIFT detector.

    Attributes:
        max_features: How many keypoints to keep, strongest first. Matching cost grows with
            the square of this.
        n_octave_layers: How finely scale is sampled. More layers find more keypoints and
            cost more. Lowe's paper uses 3.
        contrast_threshold: How strong a keypoint must be. Raising it drops weak keypoints
            in flat or dark areas. OpenCV divides it by `n_octave_layers`, so the two
            interact.
        edge_threshold: Rejects keypoints lying on an edge rather than a corner - those slide
            along the edge and are poorly localised. Larger keeps more of them.
        sigma: Blur assumed to be already present in the image. Lowe's paper uses 1.6.
    """

    model_config = ConfigDict(extra='forbid')

    max_features: int = Field(default=1000, gt=0)
    n_octave_layers: int = Field(default=3, ge=1)
    contrast_threshold: float = Field(default=0.04, gt=0.0)
    edge_threshold: float = Field(default=10.0, gt=0.0)
    sigma: float = Field(default=1.6, gt=0.0)


@FEATURE_DETECTOR_CATALOG.register('sift')
class SiftFeatureDetector(FeatureDetector):
    """
    SIFT detector and descriptor.

    Scale invariant, with a 128 dimensional float descriptor compared under the L2 norm.
    """
    produces_descriptors: ClassVar[bool] = True
    descriptor_norm: ClassVar[Optional[DescriptorNorm]] = 'l2'

    def __init__(self, config: SiftFeatureDetectorConfig):
        """
        Args:
            config: SIFT config
        """
        self._sift = cv2.SIFT_create(
            nfeatures=config.max_features,
            nOctaveLayers=config.n_octave_layers,
            contrastThreshold=config.contrast_threshold,
            edgeThreshold=config.edge_threshold,
            sigma=config.sigma
        )

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray]]:
        keypoints, descriptors = self._sift.detectAndCompute(frame, None)
        return pack_keypoints(keypoints, descriptors)
