"""
Affine warp utilities.

A "warp" is always a 2x3 affine matrix `[A | t]` mapping a point `p` to `A @ p + t`.

Two coordinate spaces are used throughout the CMC module:

- *pixel* space, where coordinates are in `[0, W] x [0, H]`. Estimators that work on
  images (feature matching, optical flow) naturally produce warps in this space.
- *normalized* space, where coordinates are in `[0, 1]^2`. Bounding boxes and motion
  filter states are stored normalized, so every CMC algorithm must return its warp in
  this space.

`pixel_warp_to_normalized` converts between the two. Note that it is NOT enough to
rescale the translation column: the linear block has to be conjugated as well, otherwise
rotation and shear are silently wrong on non-square images.
"""
from typing import Tuple

import numpy as np


def identity_warp(dtype: np.dtype = np.float32) -> np.ndarray:
    """
    Creates an identity affine warp.

    Args:
        dtype: Matrix dtype

    Returns:
        Identity 2x3 affine matrix
    """
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=dtype)


def is_identity_warp(warp: np.ndarray, atol: float = 1e-8) -> bool:
    """
    Checks whether a warp is (numerically) the identity transformation.

    Args:
        warp: Affine 2x3 matrix
        atol: Absolute tolerance

    Returns:
        True if the warp is an identity transformation
    """
    return bool(np.allclose(warp, identity_warp(dtype=warp.dtype), atol=atol))


def apply_warp_to_points(warp: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Applies an affine warp to a set of points.

    Args:
        warp: Affine 2x3 matrix
        points: Points of shape (N, 2)

    Returns:
        Warped points of shape (N, 2)
    """
    assert points.ndim == 2 and points.shape[1] == 2, f'Expected points of shape (N, 2) but got {points.shape}!'
    return points @ warp[:, :2].T + warp[:, 2]


def compose_warps(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """
    Composes two warps into a single one. The result applies `first` and then `second`,
    i.e. it is equivalent to `second o first`.

    Args:
        first: Affine 2x3 matrix applied first
        second: Affine 2x3 matrix applied second

    Returns:
        Composed affine 2x3 matrix
    """
    linear = second[:, :2] @ first[:, :2]
    translation = second[:, :2] @ first[:, 2] + second[:, 2]
    return np.concatenate([linear, translation[:, None]], axis=1).astype(first.dtype)


def invert_warp(warp: np.ndarray) -> np.ndarray:
    """
    Inverts an affine warp.

    Args:
        warp: Affine 2x3 matrix

    Returns:
        Inverted affine 2x3 matrix

    Raises:
        numpy.linalg.LinAlgError: If the linear block is singular.
    """
    linear_inv = np.linalg.inv(warp[:, :2])
    translation_inv = -linear_inv @ warp[:, 2]
    return np.concatenate([linear_inv, translation_inv[:, None]], axis=1).astype(warp.dtype)


def pixel_warp_to_normalized(warp: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Converts a warp expressed in pixel coordinates into an equivalent warp expressed in
    normalized [0, 1] coordinates.

    With `S = diag(1 / width, 1 / height)` the normalized warp is `[S A S^-1 | S t]`.
    Written out, the off-diagonal terms of the linear block pick up the image aspect ratio:

        A' = [[a00,                 a01 * height / width],
              [a10 * width / height, a11                ]]
        t' = [t0 / width, t1 / height]

    Pure translation and isotropic scale are therefore left unchanged, but rotation and
    shear are not - which is why rescaling only the translation column is incorrect.

    Note that this conversion is invariant to a uniform rescaling of the image: estimating
    a warp on a downscaled frame and normalizing by that frame's dimensions yields the
    same normalized warp as estimating at full resolution.

    Args:
        warp: Affine 2x3 matrix in pixel coordinates
        width: Image width the warp was estimated on
        height: Image height the warp was estimated on

    Returns:
        Affine 2x3 matrix in normalized coordinates
    """
    return _rescale_warp(warp, width=width, height=height, to_normalized=True)


def normalized_warp_to_pixel(warp: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Inverse of `pixel_warp_to_normalized`.

    Args:
        warp: Affine 2x3 matrix in normalized coordinates
        width: Target image width
        height: Target image height

    Returns:
        Affine 2x3 matrix in pixel coordinates
    """
    return _rescale_warp(warp, width=width, height=height, to_normalized=False)


def _rescale_warp(warp: np.ndarray, width: int, height: int, to_normalized: bool) -> np.ndarray:
    """
    Shared implementation of the pixel <-> normalized warp conversion.

    Args:
        warp: Affine 2x3 matrix
        width: Image width
        height: Image height
        to_normalized: Convert pixel -> normalized if True, else normalized -> pixel

    Returns:
        Converted affine 2x3 matrix
    """
    assert warp.shape == (2, 3), f'Expected a 2x3 affine matrix but got {warp.shape}!'
    assert width > 0 and height > 0, f'Invalid image size ({width}, {height})!'

    scale_x, scale_y = (width, height) if to_normalized else (height, width)

    result = warp.astype(np.float64).copy()
    result[0, 1] *= scale_y / scale_x
    result[1, 0] *= scale_x / scale_y
    if to_normalized:
        result[0, 2] /= width
        result[1, 2] /= height
    else:
        result[0, 2] *= width
        result[1, 2] *= height

    return result.astype(warp.dtype)


def blend_with_identity(warp: np.ndarray, weight: float) -> np.ndarray:
    """
    Linearly interpolates between the identity warp and `warp`, used to damp a correction.

    This is a plain matrix interpolation rather than a proper interpolation on the affine
    group. It is only meaningful for warps that are close to the identity, which is the
    regime CMC operates in.

    Args:
        warp: Affine 2x3 matrix
        weight: Blending weight, 0 gives the identity and 1 gives `warp`

    Returns:
        Blended affine 2x3 matrix
    """
    assert 0.0 <= weight <= 1.0, f'Blending weight must be in [0, 1] but got {weight}!'
    return ((1.0 - weight) * identity_warp(dtype=np.float64) + weight * warp.astype(np.float64)).astype(warp.dtype)


def image_size_from_frame(frame: np.ndarray) -> Tuple[int, int]:
    """
    Extracts (width, height) from a frame.

    Args:
        frame: Image of shape (H, W, C) or (H, W)

    Returns:
        Image (width, height)
    """
    return int(frame.shape[1]), int(frame.shape[0])
