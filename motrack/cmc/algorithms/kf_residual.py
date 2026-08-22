"""
Camera motion compensation from Kalman filter residuals. Uses no images.

Steps:
1. Take the motion model predictions for the current frame (`tracklet_bbox_predictions`).
2. Associate them with the current detections, reusing the tracker's association algorithm.
3. Each matched pair is a correspondence: prediction -> detection.
4. Estimate the warp from those correspondences.

The prediction is where the object would be under the previous frame's camera, so the residual
against its detection is camera motion. Object motion needs no subtraction - the prediction
already carries it:
```
prior = x[t-1] + v          # old camera frame
z     = W(prior)            # new camera frame
```

Everything is in normalized coordinates, so there is no pixel space stage and no conversion.
"""
import logging
from typing import List, Literal, Optional, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from motrack.cmc.algorithms.base import CameraMotionCompensation, CMCContext
from motrack.cmc.catalog import CMC_CATALOG
from motrack.cmc.components.ransac import WarpRANSACEstimator
from motrack.cmc.components.warp import identity_warp
from motrack.library.cv.bbox import PredBBox

logger = logging.getLogger(__name__)

# Three points give six equations for six unknowns: an exact affine fit with no redundancy.
MIN_AFFINE_CORRESPONDENCES = 3


class AssociationConfig(BaseModel):
    """
    Configuration for the association that pairs predictions with detections.

    Attributes:
        name: Association algorithm, from the tracker's own catalog.
        params: Algorithm parameters.
    """

    model_config = ConfigDict(extra='forbid')

    name: str = 'iou'
    params: dict = Field(default_factory=lambda: {'match_threshold': 0.30})


class RANSACConfig(BaseModel):
    """
    Configuration for the RANSAC algorithm.

    Attributes:
        enabled: Whether to estimate the warp robustly.
        residual_threshold: In normalized units, unlike the image based algorithms. 0.01 is
            roughly 10 px at the 960 px working resolution. Looser than their 3 px because a
            residual here carries motion model error, not sub-pixel optical flow.
        min_inliers: Lower than the image based default, because there are only as many
            correspondences as tracked objects.
        max_iterations: Max RANSAC iterations.
        max_skips: Max consecutive degenerate samples before giving up.
    """

    model_config = ConfigDict(extra='forbid')

    enabled: bool = True
    residual_threshold: float = 0.01
    max_iterations: int = 500
    min_inliers: int = 4
    max_skips: int = 100


class KFResidualCMCConfig(BaseModel):
    """
    Configuration for the KFResidualCMC algorithm.

    Attributes:
        detection_threshold: Drop detections below this confidence before associating.
        association: Which association algorithm pairs predictions with detections.
        motion_model: `translation` fits two parameters from the median displacement,
            `translation-mean` the same two from the mean, `affine` fits six via RANSAC. Affine
            needs redundancy this algorithm rarely has, and the mean is there to show what the
            median's robustness is worth.
        points: `center` takes one point per match, `corners` takes four. Corners are the only
            way scale is observed, and they raise the correspondence count - but the four are
            determined by two numbers, so RANSAC counts correlated points as independent.
        min_correspondences: Below this, return identity rather than estimate.
        ransac: RANSAC configuration, used by the affine model only.
        seed: RANSAC seed.
    """

    model_config = ConfigDict(extra='forbid')

    detection_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    association: AssociationConfig = Field(default_factory=AssociationConfig)
    motion_model: Literal['translation', 'translation-mean', 'affine'] = 'translation'
    points: Literal['center', 'corners'] = 'center'
    min_correspondences: int = Field(default=3, ge=1)
    ransac: RANSACConfig = Field(default_factory=RANSACConfig)

    seed: int = Field(default=42)


CMC_CATALOG.register_config('kf-residual')(KFResidualCMCConfig)


@CMC_CATALOG.register('kf-residual')
class KFResidualCMC(CameraMotionCompensation):
    """
    Camera motion compensation from the residual between motion model predictions and detections.

    Reads no frames, so it costs a fraction of the image based algorithms.
    """

    requires_image: bool = False

    def __init__(self, config: KFResidualCMCConfig):
        """
        Args:
            config: Configuration for the KFResidualCMC algorithm.
        """
        # Imported here rather than at module scope: `motrack.tracker` imports `motrack.cmc`,
        # so pulling the association factory in at the top closes an import cycle.
        from motrack.tracker.matching.factory import association_factory  # pylint: disable=import-outside-toplevel

        self._config = config
        self._association = association_factory(config.association.name, config.association.params)

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

    def apply(self, ctx: CMCContext) -> np.ndarray:
        """
        Applies the KFResidualCMC algorithm to the given context.

        Steps:
        1. Take the motion model predictions for the current frame.
        2. Associate them with the detections.
        3. Build correspondences from the matched boxes.
        4. Estimate the warp.
        """
        predictions = ctx.tracklet_bbox_predictions
        if not predictions or not ctx.detections:
            # Either no tracklets yet, or no detections this frame.
            return identity_warp()

        src, dst = self._correspondences(predictions, ctx.detections)
        if len(src) < self._config.min_correspondences:
            logger.debug(f'Only {len(src)} correspondences in frame {ctx.frame_index}.')
            return identity_warp()

        return self._estimate(src, dst)

    def _correspondences(
        self,
        predictions: List[PredBBox],
        detections: List[PredBBox]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pairs predictions with detections and returns their points.

        Args:
            predictions: Motion model predictions for the current frame
            detections: Current frame detections

        Returns:
            Source and target points, both (N, 2) in normalized coordinates
        """
        kept = [d for d in detections if d.conf is None or d.conf >= self._config.detection_threshold]
        if len(predictions) == 0 or len(kept) == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

        matches, _, _ = self._association.match(predictions, kept)
        if len(matches) == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

        src = np.concatenate([self._points(predictions[t]) for t, _ in matches])
        dst = np.concatenate([self._points(kept[d]) for _, d in matches])
        return src, dst

    def _points(self, bbox: PredBBox) -> np.ndarray:
        """
        Turns one box into the points it contributes.

        Args:
            bbox: Bounding box, in normalized coordinates

        Returns:
            Points, (1, 2) for `center` or (4, 2) for `corners`
        """
        if self._config.points == 'center':
            return np.array([[bbox.center.x, bbox.center.y]], dtype=np.float32)

        left, top = bbox.upper_left.x, bbox.upper_left.y
        right, bottom = bbox.bottom_right.x, bbox.bottom_right.y
        return np.array([[left, top], [right, top], [left, bottom], [right, bottom]], dtype=np.float32)

    def _estimate(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """
        Fits the configured motion model to the correspondences.

        Args:
            src: Source points (N, 2), normalized
            dst: Target points (N, 2), normalized

        Returns:
            Affine 2x3 matrix in normalized coordinates, identity when estimation fails
        """
        if self._config.motion_model in ('translation', 'translation-mean'):
            # The median is already robust, so translation needs no RANSAC. The mean is offered
            # only to measure what that robustness buys: one badly matched pair moves it.
            reduce = np.median if self._config.motion_model == 'translation' else np.mean
            warp = identity_warp()
            warp[:, 2] = reduce(dst - src, axis=0)
            return warp

        if len(src) < MIN_AFFINE_CORRESPONDENCES or self._ransac is None:
            return identity_warp()

        estimate = self._ransac.estimate(src, dst)
        return estimate.warp if estimate.success else identity_warp()
