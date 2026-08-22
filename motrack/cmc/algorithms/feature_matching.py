"""
Camera motion compensation based on descriptor matching between two frames.

Steps:
1. Use feature detector (e.g. ORB/SIFT) to detect features in both frames.
2. [Optional] Use detections to exclude features from the estimation.
3. Match the features by descriptor, using the norm the detector declares.
4. [Optional] Reject correspondences whose displacement is implausible.
5. Calculate the warp matrix using the RANSAC algorithm.

Features are matched rather than tracked, so the detector has to produce descriptors. That
rules out Shi-Tomasi, which yields corner locations only - it can be tracked but not matched.
"""
import logging
from typing import List, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from motrack.cmc.algorithms.base import CMCContext, CameraMotionCompensation
from motrack.cmc.algorithms.utils import (
    estimate_normalized_warp,
    exclude_points_in_detections,
    to_grayscale,
)
from motrack.cmc.catalog import CMC_CATALOG
from motrack.cmc.components.feature_detector.factory import feature_detector_factory
from motrack.cmc.components.matching import match_descriptors
from motrack.cmc.components.ransac import WarpRANSACEstimator
from motrack.cmc.components.spatial_filter import filter_by_displacement
from motrack.library.cv.bbox import PredBBox
from motrack.cmc.components.warp import identity_warp

logger = logging.getLogger(__name__)


class FeatureDetectorConfig(BaseModel):
    """
    Configuration for the feature detector.
    """

    model_config = ConfigDict(extra='forbid')

    type: str = 'orb'
    params: dict = Field(default_factory=dict)


class MatchingConfig(BaseModel):
    """
    Configuration for descriptor matching.

    Attributes:
        ratio_threshold: Keep a match when the best candidate is closer than this fraction of
            the second best. Lower is stricter. BoT-SORT uses 0.9, Lowe's SIFT paper suggests
            0.8; the permissive value works because RANSAC rejects the remaining outliers.
    """

    model_config = ConfigDict(extra='forbid')

    ratio_threshold: float = Field(default=0.9, gt=0.0, le=1.0)


class ExclusionConfig(BaseModel):
    """
    Configuration for excluding features that land on detected objects.

    The bounding boxes are not configured here: they are the current frame's detections and
    change every frame, so they arrive through the context.

    Attributes:
        enabled: Whether to exclude features that land on a detection.
        expansion_factor: How much to grow each detection before testing. A bounding box
            rarely covers the whole object, so a feature just outside one often still sits
            on it. 0 keeps exclusion on without expanding.
    """

    model_config = ConfigDict(extra='forbid')

    enabled: bool = False
    expansion_factor: float = Field(default=0.2, ge=0.0)


class SpatialFilterConfig(BaseModel):
    """
    Configuration for the BoT-SORT geometric heuristic applied before estimation.

    Attributes:
        enabled: Whether to apply any spatial filtering at all.
        max_relative: Reject displacements above this fraction of the frame. A sanity bound
            that does not depend on the other correspondences.
        n_std: Keep displacements within this many standard deviations of the mean. None
            disables that stage. It overlaps with what RANSAC and detection exclusion already
            do, so it is off by default and exists to be measured.
    """

    model_config = ConfigDict(extra='forbid')

    enabled: bool = False
    max_relative: Optional[float] = Field(default=0.25, gt=0.0)
    n_std: Optional[float] = Field(default=None, gt=0.0)


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


class FeatureMatchingCMCConfig(BaseModel):
    """
    Configuration for the FeatureMatchingCMC algorithm.
    """

    model_config = ConfigDict(extra='forbid')

    feature_detector: FeatureDetectorConfig = Field(default_factory=FeatureDetectorConfig)
    matching: MatchingConfig = Field(default_factory=MatchingConfig)
    exclusion: ExclusionConfig = Field(default_factory=ExclusionConfig)
    spatial_filter: SpatialFilterConfig = Field(default_factory=SpatialFilterConfig)
    ransac: RANSACConfig = Field(default_factory=RANSACConfig)

    seed: int = Field(default=42)


CMC_CATALOG.register_config('feature-matching')(FeatureMatchingCMCConfig)


@CMC_CATALOG.register('feature-matching')
class FeatureMatchingCMC(CameraMotionCompensation):
    """
    Camera motion compensation using descriptor matching.

    Flexible in terms of feature detector, as long as it produces descriptors.
    """
    def __init__(self, config: FeatureMatchingCMCConfig):
        """
        Args:
            config: Configuration for the FeatureMatchingCMC algorithm.

        Raises:
            ValueError: If the configured detector produces no descriptors.
        """
        self._config = config
        self._feature_detector = feature_detector_factory(config.feature_detector.type, config.feature_detector.params)

        if not self._feature_detector.produces_descriptors:
            raise ValueError(
                f'Feature detector "{config.feature_detector.type}" produces no descriptors, so its features '
                f'cannot be matched. Use a detector such as "orb" or "sift", or track the features instead '
                f'with the "pylk" camera motion compensation.'
            )

        self._ransac: Optional[WarpRANSACEstimator] = None
        if config.ransac.enabled:
            self._ransac = WarpRANSACEstimator(
                residual_threshold=config.ransac.residual_threshold,
                max_iterations=config.ransac.max_iterations,
                min_inliers=config.ransac.min_inliers,
                max_skips=config.ransac.max_skips,
                seed=config.seed
            )

    def reset(self) -> None:
        if self._ransac is not None:
            self._ransac.reset()

    def _detect(
        self,
        frame: np.ndarray,
        detections: Optional[List[PredBBox]],
        image_size: tuple[int, int]
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Detects features and drops the ones that land on a detected object.

        Points and descriptors are filtered together, so they stay aligned.

        Args:
            frame: Frame to detect features in
            detections: Current frame detections, in normalized coordinates
            image_size: Frame (width, height)

        Returns:
            Points and their descriptors
        """
        points, descriptors = self._feature_detector.detect(to_grayscale(frame))
        if descriptors is None or points.shape[0] == 0:
            return points, descriptors

        if self._config.exclusion.enabled:
            keep = exclude_points_in_detections(
                points, detections, image_size, self._config.exclusion.expansion_factor
            )
            points, descriptors = points[keep], descriptors[keep]

        return points, descriptors

    def apply(self, ctx: CMCContext) -> np.ndarray:
        """
        Applies the FeatureMatchingCMC algorithm to the given context.

        Steps:
        1. Use feature detector (e.g. ORB/SIFT) to detect features in both frames.
        2. [Optional] Use detections to exclude features from the estimation.
        3. Match the features by descriptor, using the norm the detector declares.
        4. [Optional] Reject correspondences whose displacement is implausible.
        5. Calculate the warp matrix using the RANSAC algorithm.
        """
        # The tracker supplies the previous frame. Its absence means the first frame of a
        # scene or a gap in the sequence, so there is nothing to compare against.
        if ctx.prev_frame is None or ctx.curr_frame is None:
            return identity_warp()

        # 1) + 2) Detect in both frames, excluding features on detected objects.
        prev_points, prev_descriptors = self._detect(ctx.prev_frame, ctx.detections, ctx.image_size)
        curr_points, curr_descriptors = self._detect(ctx.curr_frame, ctx.detections, ctx.image_size)

        if prev_descriptors is None or curr_descriptors is None:
            logger.debug(f'No descriptors available in frame {ctx.frame_index}.')
            return identity_warp()

        # 3) Match by descriptor. The norm is a property of the detector, not a free choice.
        pairs = match_descriptors(
            prev_descriptors,
            curr_descriptors,
            norm=self._feature_detector.descriptor_norm,
            ratio_threshold=self._config.matching.ratio_threshold
        )
        if pairs.shape[0] == 0:
            logger.debug(f'No descriptor matches in frame {ctx.frame_index}.')
            return identity_warp()

        src, dst = prev_points[pairs[:, 0]], curr_points[pairs[:, 1]]

        # 4) [Optional] Reject implausible displacements before estimating.
        if self._config.spatial_filter.enabled:
            keep = filter_by_displacement(
                src, dst, ctx.image_size,
                max_relative=self._config.spatial_filter.max_relative,
                n_std=self._config.spatial_filter.n_std
            )
            src, dst = src[keep], dst[keep]

        # 5) Calculate the warp matrix, in normalized coordinates.
        return estimate_normalized_warp(src, dst, ctx.image_size, self._ransac)
