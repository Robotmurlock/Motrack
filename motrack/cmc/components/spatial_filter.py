"""
Geometric filtering of correspondences before the warp is estimated. Reproduces the heuristic
BoT-SORT applies between matching and transform estimation.
Supporting module for feature based camera motion compensation.

Steps:
1. Compute the displacement of every correspondence.
2. [Optional] Reject displacements larger than a fraction of the frame.
3. [Optional] Reject displacements too far from the mean displacement.

Both stages are switchable, so each one's effect can be measured on its own.

Absolute cap: a camera cannot move a quarter of the frame between consecutive frames at
25-30 fps, so this is a sanity bound. It does not depend on the other correspondences.
```
keep = |dst - src| <= max_relative * (width, height)
```

Statistical pass: mean and std are themselves dragged by the outliers they are meant to catch,
so the threshold widens when it should tighten. RANSAC does the same job by consensus. Off by
default; kept so the ablation is expressible.
```
keep = |d - mean(d)| <= n_std * std(d)
```

BoT-SORT clips one-sided (above the mean, no absolute value), which looks unintended - the
symmetric form is used here.

Reference: https://arxiv.org/abs/2206.14651, heuristic in `tracker/gmc.py`
"""
from typing import Optional, Tuple

import numpy as np


def filter_by_displacement(
    src: np.ndarray,
    dst: np.ndarray,
    image_size: Tuple[int, int],
    max_relative: Optional[float] = 0.25,
    n_std: Optional[float] = None
) -> np.ndarray:
    """
    Rejects correspondences whose displacement is implausible for camera motion.

    Args:
        src: Source points in pixel coordinates, shape (N, 2)
        dst: Target points in pixel coordinates, shape (N, 2)
        image_size: Frame (width, height)
        max_relative: Reject a displacement exceeding this fraction of the frame width or
            height. None disables the cap.
        n_std: Keep displacements within this many standard deviations of the mean, per axis.
            None disables the statistical pass, which is the default - RANSAC covers it.

    Returns:
        Boolean mask of the correspondences to keep, shape (N,)
    """
    assert src.shape == dst.shape, f'Source and target must have the same shape! Got {src.shape} and {dst.shape}.'

    keep = np.ones(src.shape[0], dtype=bool)
    if src.shape[0] == 0:
        return keep

    displacement = dst - src

    if max_relative is not None:
        width, height = image_size
        limit = max_relative * np.array([width, height], dtype=np.float32)
        keep &= np.all(np.abs(displacement) <= limit, axis=1)

    if n_std is not None and keep.any():
        surviving = displacement[keep]
        mean = np.mean(surviving, axis=0)
        deviation = np.std(surviving, axis=0)

        # A zero deviation means every surviving displacement is identical, so nothing
        # deviates and the whole set is kept.
        within = np.all(np.abs(displacement - mean) <= n_std * deviation, axis=1) | np.all(deviation == 0)
        keep &= within

    return keep
