"""
Helpers shared by the correspondence based camera motion compensation algorithms.

`PyLKCMC` and `FeatureMatchingCMC` differ only in how they produce correspondences: one
tracks points with optical flow, the other matches descriptors. Everything either side of
that - excluding points that fall on detected objects, and turning correspondences into a
normalized warp - is the same, and lives here.
"""
from typing import List, Optional

import cv2
import numpy as np

from motrack.cmc.components.ransac import WarpRANSACEstimator, estimate_warp_lstsq
from motrack.cmc.components.warp import identity_warp, pixel_warp_to_normalized
from motrack.library.cv.bbox import PredBBox


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    """
    Converts a frame to grayscale, passing single channel input through unchanged.

    Frames arrive RGB, not BGR, so a BGR conversion here would quietly degrade keypoint
    repeatability without ever failing.

    Args:
        frame: RGB or grayscale frame

    Returns:
        Grayscale frame
    """
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame


def exclude_points_in_detections(
    points: np.ndarray,
    detections: Optional[List[PredBBox]],
    image_size: tuple[int, int],
    expansion_factor: float
) -> np.ndarray:
    """
    Drops points that fall inside a detected object.

    Detections describe things that move independently of the camera, so correspondences on
    them do not describe camera motion. Their bounding boxes are expanded first, because a
    box rarely covers a whole object and a feature just outside one still sits on it.

    Detections are in normalized coordinates while points are in pixels, so the points are
    normalized rather than the boxes scaled - that keeps the comparison in the coordinate
    space the boxes are already defined in.

    Args:
        points: Points in pixel coordinates, shape (N, 2)
        detections: Current frame detections, in normalized coordinates
        image_size: Frame (width, height)
        expansion_factor: Relative amount to grow each box by before testing

    Returns:
        Boolean mask of the points to keep, shape (N,)
    """
    keep = np.ones(points.shape[0], dtype=bool)
    if not detections or points.shape[0] == 0:
        return keep

    width, height = image_size
    normalized = points / np.array([width, height], dtype=np.float32)

    for detection in detections:
        bbox = detection.expand(expansion_factor, clip=True) if expansion_factor > 0 else detection
        inside = (
            (normalized[:, 0] >= bbox.upper_left.x)
            & (normalized[:, 0] <= bbox.bottom_right.x)
            & (normalized[:, 1] >= bbox.upper_left.y)
            & (normalized[:, 1] <= bbox.bottom_right.y)
        )
        keep &= ~inside

    return keep


def estimate_normalized_warp(
    src: np.ndarray,
    dst: np.ndarray,
    image_size: tuple[int, int],
    ransac: Optional[WarpRANSACEstimator]
) -> np.ndarray:
    """
    Turns pixel space correspondences into a warp in normalized coordinates.

    Bounding boxes and motion filter states are normalized, so the warp has to be too -
    returning a pixel space warp would scale every translation by the frame dimensions.

    Args:
        src: Source points in pixel coordinates, shape (N, 2)
        dst: Target points in pixel coordinates, shape (N, 2)
        image_size: Frame (width, height)
        ransac: Robust estimator, or None to use a plain least-squares fit over every
            correspondence. The plain fit exists to quantify what robust estimation buys.

    Returns:
        Affine 2x3 matrix in normalized coordinates, identity when estimation fails
    """
    width, height = image_size

    if ransac is not None:
        estimate = ransac.estimate(src, dst)
        if not estimate.success:
            return identity_warp()
        warp = estimate.warp
    else:
        warp, is_degenerate = estimate_warp_lstsq(src, dst)
        if is_degenerate:
            return identity_warp()

    return pixel_warp_to_normalized(warp, width=width, height=height)
