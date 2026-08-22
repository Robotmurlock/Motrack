"""
Affine warp estimation from point correspondences.

Two estimators are provided:

- `estimate_warp_lstsq`: plain least-squares fit over all given correspondences.
- `WarpRANSACEstimator`: robust fit that tolerates outlier correspondences, followed by a
  final least-squares refit over the identified inliers.

Both are **unit-agnostic**: `src`/`dst` may be in pixels, in normalized [0, 1] coordinates,
or in anything else, as long as the residual threshold is expressed in the same units.
Feature based CMC works in (downscaled) pixels, while the Kalman filter residual CMC works
directly in normalized coordinates.
"""
from dataclasses import dataclass
import logging
from typing import Tuple

import numpy as np

from motrack.cmc.components.warp import apply_warp_to_points, identity_warp

logger = logging.getLogger(__name__)

MIN_SAMPLES = 3


@dataclass(frozen=True)
class AffineEstimate:
    """
    Result of a robust affine warp estimation.

    Attributes:
        warp: Affine 2x3 matrix. Identity when `success` is False.
        inliers_mask: Boolean mask over the input correspondences. All False on failure.
        n_inliers: Number of inliers supporting `warp`.
        n_iterations: Number of samples drawn.
        n_degenerate: Number of samples rejected as degenerate (duplicate/collinear points).
        success: Whether a warp supported by at least `min_inliers` correspondences was found.
    """
    warp: np.ndarray
    inliers_mask: np.ndarray
    n_inliers: int
    n_iterations: int
    n_degenerate: int
    success: bool

    @property
    def inlier_ratio(self) -> float:
        """
        Returns:
            Fraction of the input correspondences that support the estimated warp
        """
        n_total = self.inliers_mask.shape[0]
        return self.n_inliers / n_total if n_total > 0 else 0.0


def estimate_warp_lstsq(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Estimates an affine warp between two sets of points using least-squares.

    Example:
        src = [
            [0, 0],
            [1, 0],
            [0, 1],
        ]
        dst = [
            [1, -1],
            [3, 0],
            [2, -1],
        ]
        -> warp = [[2, 1, 1], [1, 0, -1]]

    Args:
        src: Source points (shape: (N, 2))
        dst: Target points (shape: (N, 2))

    Returns:
        Affine warp matrix, bool indicating if the warp is degenerate
    """
    assert src.shape == dst.shape, f'Source and target points must have the same shape! Got {src.shape} and {dst.shape}.'
    assert src.ndim == 2 and src.shape[1] == 2, f'Expected points of shape (N, 2) but got {src.shape}!'

    if src.shape[0] < MIN_SAMPLES:
        return identity_warp(), True

    src_expanded = np.hstack([src, np.ones((src.shape[0], 1))]).astype(np.float64)

    solution, _, rank, _ = np.linalg.lstsq(src_expanded, dst.astype(np.float64), rcond=None)
    if rank < MIN_SAMPLES:
        return identity_warp(), True

    return solution.T.astype(np.float32), False


class WarpRANSACEstimator:
    """
    RANSAC estimator for affine warp. Algorithm:

    Until a stopping criterion is met:
        1. Sample a minimal set of points
        2. Estimate a warp using the sampled points and count number of inliers over all points.
            If the number of inliers beats the current best, update the best inliers mask.
            Otherwise, skip the iteration.
        3. If the number of skips exceeds the max number of skips, stop the algorithm.

    Stop criterion:
        - Number of iterations exceeds the max number of iterations
        - Number of skips exceeds the max number of skips

    Refit the warp over the best inliers mask and return it, otherwise return the identity warp.
    """

    def __init__(
        self,
        residual_threshold: float = 5.0,
        max_iterations: int = 100,
        min_inliers: int = 10,
        max_skips: int = 10,
        seed: int = 42
    ) -> None:
        """
        Args:
            residual_threshold: Threshold for inlier detection, in input point units (float)
            max_iterations: Maximum number of iterations (int)
            min_inliers: Minimum number of inliers required for a valid warp (int)
            max_skips: Maximum number of skips allowed (int)
            seed: Seed for the random number generator (int)
        """
        assert residual_threshold > 0, f'Residual threshold must be positive but got {residual_threshold}!'
        assert min_inliers >= MIN_SAMPLES, f'At least {MIN_SAMPLES} inliers are required but got {min_inliers}!'
        assert max_iterations >= 1, f'At least one iteration is required but got {max_iterations}!'
        assert max_skips >= 1, f'At least one skip is required but got {max_skips}!'

        self._residual_threshold = residual_threshold
        self._max_iterations = max_iterations
        self._min_inliers = min_inliers
        self._max_skips = max_skips

        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        """
        Re-seeds the random number generator.

        Called between scenes so that a scene's results do not depend on how many samples
        the preceding scenes happened to consume.
        """
        self._rng = np.random.default_rng(self._seed)

    def _failure(self, n_points: int, n_iterations: int, n_degenerate: int) -> AffineEstimate:
        """
        Creates a failed estimate carrying an identity warp.

        Args:
            n_points: Number of input correspondences
            n_iterations: Number of samples drawn
            n_degenerate: Number of degenerate samples

        Returns:
            Failed affine estimate
        """
        return AffineEstimate(
            warp=identity_warp(),
            inliers_mask=np.zeros(n_points, dtype=bool),
            n_inliers=0,
            n_iterations=n_iterations,
            n_degenerate=n_degenerate,
            success=False
        )

    def _score(self, warp: np.ndarray, src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Scores a warp against all correspondences.

        Args:
            warp: Affine 2x3 matrix
            src: Source points (shape: (N, 2))
            dst: Target points (shape: (N, 2))

        Returns:
            Inlier mask and inlier count
        """
        residuals = np.linalg.norm(apply_warp_to_points(warp, src) - dst, axis=1)
        inliers_mask = residuals < self._residual_threshold
        return inliers_mask, int(np.sum(inliers_mask))

    def estimate(self, src: np.ndarray, dst: np.ndarray) -> AffineEstimate:
        """
        Estimates an affine warp between two sets of points using RANSAC.

        Args:
            src: Source points (shape: (N, 2))
            dst: Target points (shape: (N, 2))

        Returns:
            Affine estimate
        """
        # Validation
        assert src.shape == dst.shape, f'Source and target points must have the same shape! Got {src.shape} and {dst.shape}.'
        assert src.ndim == 2 and src.shape[1] == 2, f'Expected points of shape (N, 2) but got {src.shape}!'

        n_points = src.shape[0]
        if n_points < self._min_inliers:
            logger.debug('Not enough points to estimate warp! Got %d points, but %d are required.', n_points, self._min_inliers)
            return self._failure(n_points, n_iterations=0, n_degenerate=0)

        # State
        best_inliers_mask = None
        best_n_inliers = 0

        n_skips = 0
        iteration = 0

        # Stats
        n_degenerate = 0

        # Main loop
        while iteration < self._max_iterations and n_skips < self._max_skips:
            iteration += 1

            # 1. Sample a minimal set of points
            sampled_indices = self._rng.choice(n_points, size=MIN_SAMPLES, replace=False)
            warp, is_degenerate = estimate_warp_lstsq(src[sampled_indices], dst[sampled_indices])
            if is_degenerate:
                n_degenerate += 1
                n_skips += 1
                continue

            # 2. Count number of inliers over all points
            inliers_mask, n_inliers = self._score(warp, src, dst)

            # 3. If the number of inliers beats the current best, update the best inliers mask
            if n_inliers > best_n_inliers:
                best_inliers_mask = inliers_mask
                best_n_inliers = n_inliers
                n_skips = 0
            else:
                n_skips += 1

        # 4. Refit the warp over the best inliers mask
        if best_inliers_mask is None or best_n_inliers < self._min_inliers:
            logger.debug('No valid warp found after %d iterations!', iteration)
            return self._failure(n_points, n_iterations=iteration, n_degenerate=n_degenerate)

        refit_warp, is_degenerate = estimate_warp_lstsq(src[best_inliers_mask], dst[best_inliers_mask])
        if is_degenerate:
            logger.debug('Best inliers mask of %d points is degenerate!', best_n_inliers)
            return self._failure(n_points, n_iterations=iteration, n_degenerate=n_degenerate)

        refit_inliers_mask, refit_n_inliers = self._score(refit_warp, src, dst)
        if refit_n_inliers < self._min_inliers:
            logger.debug('Refit dropped below %d inliers (%d)!', self._min_inliers, refit_n_inliers)
            return self._failure(n_points, n_iterations=iteration, n_degenerate=n_degenerate)

        return AffineEstimate(
            warp=refit_warp,
            inliers_mask=refit_inliers_mask,
            n_inliers=refit_n_inliers,
            n_iterations=iteration,
            n_degenerate=n_degenerate,
            success=True
        )
