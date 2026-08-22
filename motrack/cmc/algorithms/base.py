"""
Camera motion compensation interface.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, List, Optional, Tuple

import numpy as np

from motrack.library.cv.bbox import PredBBox


@dataclass(frozen=True)
class CMCContext:
    """
    Per-frame tracker state made available to a CMC algorithm.

    The field set is closed rather than open-ended: at the point CMC runs there are exactly
    three sources of information in the tracker - the raw image, the detector output and
    the motion model output - plus the identifiers needed to address a frame.

    Attributes:
        frame_index: Zero-based index of the *current* frame.
        scene: Scene name, when the tracker is running over a named scene.
        prev_frame: Previous frame image (RGB, HxWx3). The tracker caches it, so algorithms
            do not have to. It is None on the first frame of a scene, after a gap in the
            frame sequence, and when image loading is disabled - in every one of those cases
            there is nothing to compare against and `apply` returns an identity warp.
        curr_frame: Current frame image (RGB, HxWx3). None when image loading is disabled.
        image_size: Current frame (width, height). Always set when `curr_frame` is set.
        detections: Current frame detections in normalized coordinates. These are the raw
            detector outputs, before the tracker applies its own detection threshold.
        tracklet_bbox_predictions: Motion model predictions for the current frame, in
            normalized coordinates, before this warp is applied to them.
    """
    frame_index: int
    scene: Optional[str] = None
    prev_frame: Optional[np.ndarray] = None
    curr_frame: Optional[np.ndarray] = None
    image_size: Optional[Tuple[int, int]] = None
    detections: Optional[List[PredBBox]] = None
    tracklet_bbox_predictions: Optional[List[PredBBox]] = None


class CameraMotionCompensation(ABC):
    """
    Camera motion compensation interface.

    Attributes:
        requires_image: Whether the algorithm needs the context frames to be set. Trackers
            use this to fail fast when image loading is disabled.
    """
    requires_image: ClassVar[bool] = True

    @abstractmethod
    def apply(self, ctx: CMCContext) -> np.ndarray:
        """
        Estimates the affine transformation the camera applied to the image between the
        previous and the current frame.

        The returned warp maps normalized [0, 1] coordinates expressed in frame
        `ctx.frame_index - 1` into normalized coordinates of frame `ctx.frame_index`.

        Implementations must never raise on degenerate input. When the transformation cannot
        be estimated an identity warp is returned instead. The first frame of a scene and a
        gap in the frame sequence are both signalled by `ctx.prev_frame` being None, so an
        algorithm only has to handle that one case plus its own failures, such as too few
        correspondences.

        Args:
            ctx: Current frame context

        Returns:
            Affine 2x3 matrix (includes translation) in normalized coordinates
        """

    def reset(self) -> None:
        """
        Drops per-scene state. Called by the tracker before every scene.

        Stateless algorithms do not need to override this.
        """
