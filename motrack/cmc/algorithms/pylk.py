"""
Camera motion compensation based on the PyLucasKanade algorithm for optical flow estimation.

Steps:
1. Use feature detector (e.g. Shi-Tomasi) to detect features to track.
2. [Optional] Use detections to exclude features from the estimation.
3. Use PyLucasKanade algorithm to estimate the optical flow for each feature.
4. Calculate the warp matrix using the RANSAC algorithm.
"""
import logging
from typing import Literal, Optional

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from motrack.cmc.algorithms.base import CameraMotionCompensation, CMCContext
from motrack.cmc.algorithms.utils import (
    estimate_normalized_warp,
    exclude_points_in_detections,
    mask_detections_in_image,
    to_grayscale,
)
from motrack.cmc.catalog import CMC_CATALOG
from motrack.cmc.components.feature_detector.factory import feature_detector_factory
from motrack.cmc.components.pylk import PyLucasKanadeEstimator
from motrack.cmc.components.ransac import WarpRANSACEstimator
from motrack.cmc.components.warp import identity_warp

logger = logging.getLogger(__name__)


class FeatureDetectorConfig(BaseModel):
    """
    Configuration for the feature detector.
    """

    model_config = ConfigDict(extra='forbid')

    type: str = 'shi-tomasi'
    params: dict = Field(default_factory=dict)


class ExclusionConfig(BaseModel):
    """
    Configuration for excluding features that land on detected objects.

    The bounding boxes are not configured here: they are the current frame's detections and
    change every frame, so they arrive through the context.

    Attributes:
        enabled: Whether to exclude features that land on a detection.
        mode: How the exclusion is applied. `points` detects features over the whole frame and
            then discards the ones inside a detection. `image` fills the detections with black
            before detection runs, so no feature is found there. The two are not equivalent:
            filling a box introduces a step edge along its border, which corner and blob
            detectors respond to, so features removed from the object's interior reappear on
            its outline.
        expansion_factor: How much to grow each detection before testing. A bounding box
            rarely covers the whole object, so a feature just outside one often still sits
            on it. 0 keeps exclusion on without expanding.
    """

    model_config = ConfigDict(extra='forbid')

    enabled: bool = False
    mode: Literal['points', 'image'] = 'points'
    expansion_factor: float = Field(default=0.2, ge=0.0)


class PyLKConfig(BaseModel):
    """
    Configuration for the PyLucasKanade algorithm.
    """

    model_config = ConfigDict(extra='forbid')

    window_size: tuple[int, int] = (21, 21)
    max_level: int = 4
    max_iterations: int = 30
    iteration_convergence_threshold: float = 0.1


class RANSACConfig(BaseModel):
    """
    Configuration for the RANSAC algorithm.

    Attributes:
        enabled: Whether to estimate the warp robustly. Disabling it falls back to a plain
            least-squares fit over every correspondence, which exists to quantify what
            robust estimation actually buys.
    """

    model_config = ConfigDict(extra='forbid')

    enabled: bool = True
    residual_threshold: float = 3.0
    max_iterations: int = 500
    min_inliers: int = 10
    max_skips: int = 100


class PyLKCMCConfig(BaseModel):
    """
    Configuration for the PyLKCMC algorithm.

    Attributes:
    implementation: Which code path the hand-written components run through. `custom` uses the
        implementations in `motrack/cmc/components/`, which is what the report validates.
        `opencv` delegates optical flow, descriptor matching and transform estimation to OpenCV,
        so that a throughput comparison is not charged for those being written in Python. The two
        agree to well under the estimator's residual threshold (Appendix B.1).
    """

    model_config = ConfigDict(extra='forbid')

    feature_detector: FeatureDetectorConfig = Field(default_factory=FeatureDetectorConfig)
    pylk: PyLKConfig = Field(default_factory=PyLKConfig)
    exclusion: ExclusionConfig = Field(default_factory=ExclusionConfig)
    ransac: RANSACConfig = Field(default_factory=RANSACConfig)

    implementation: Literal['custom', 'opencv'] = 'custom'
    seed: int = Field(default=42)


CMC_CATALOG.register_config('pylk')(PyLKCMCConfig)


@CMC_CATALOG.register('pylk')
class PyLKCMC(CameraMotionCompensation):
    """
    Camera motion compensation using the PyLucasKanade algorithm.

    Flexible in terms of feature detector. Features are tracked rather than matched, so any
    detector works, including ones that produce no descriptors.
    """
    def __init__(self, config: PyLKCMCConfig):
        """
        Args:
            config: Configuration for the PyLKCMC algorithm.
        """
        self._config = config
        self._feature_detector = feature_detector_factory(config.feature_detector.type, config.feature_detector.params)
        self._pylk = PyLucasKanadeEstimator(
            window_size=config.pylk.window_size,
            max_level=config.pylk.max_level,
            max_iterations=config.pylk.max_iterations,
            iteration_convergence_threshold=config.pylk.iteration_convergence_threshold,
            implementation=config.implementation
        )
        self._ransac: Optional[WarpRANSACEstimator] = None
        if config.ransac.enabled:
            self._ransac = WarpRANSACEstimator(
                residual_threshold=config.ransac.residual_threshold,
                max_iterations=config.ransac.max_iterations,
                min_inliers=config.ransac.min_inliers,
                max_skips=config.ransac.max_skips,
                seed=config.seed,
                implementation=config.implementation
            )

    def reset(self) -> None:
        if self._ransac is not None:
            self._ransac.reset()

    def apply(self, ctx: CMCContext) -> np.ndarray:
        """
        Applies the PyLKCMC algorithm to the given context.

        Steps:
        1. Use feature detector (e.g. Shi-Tomasi) to detect features to track.
        2. [Optional] Use detections to exclude features from the estimation.
        3. Use PyLucasKanade algorithm to estimate the optical flow for each feature.
        4. Calculate the warp matrix using the RANSAC algorithm.
        """
        # The tracker supplies the previous frame. Its absence means the first frame of a
        # scene or a gap in the sequence, so there is nothing to compare against.
        if ctx.prev_frame is None or ctx.curr_frame is None:
            return identity_warp()

        exclusion = self._config.exclusion
        prev_frame, curr_frame = ctx.prev_frame, ctx.curr_frame

        # 1) [Optional] Fill the detections in before anything looks at the frames, so the
        # whole algorithm sees the masked images rather than only the detector.
        if exclusion.enabled and exclusion.mode == 'image':
            prev_frame = mask_detections_in_image(
                prev_frame, ctx.detections, ctx.image_size, exclusion.expansion_factor
            )
            curr_frame = mask_detections_in_image(
                curr_frame, ctx.detections, ctx.image_size, exclusion.expansion_factor
            )

        # 2) Detect features to track. Only the previous frame is needed: the features are
        # tracked forward, not matched against independently detected ones.
        prev_points, _ = self._feature_detector.detect(to_grayscale(prev_frame))
        if prev_points.shape[0] == 0:
            logger.debug('No features detected in frame %d.', ctx.frame_index)
            return identity_warp()

        # 3) [Optional] Exclude points that land on a detected object.
        if exclusion.enabled and exclusion.mode == 'points':
            keep = exclude_points_in_detections(
                prev_points, ctx.detections, ctx.image_size, exclusion.expansion_factor
            )
            prev_points = prev_points[keep]

        # 4) Estimate the optical flow for each feature.
        flow, skips = self._pylk.estimate(prev_frame, curr_frame, prev_points)

        # 5) Calculate the warp matrix using the RANSAC algorithm, in normalized coordinates.
        src, dst = prev_points[~skips], (prev_points + flow)[~skips]
        return estimate_normalized_warp(src, dst, ctx.image_size, self._ransac)
