"""
Manual comparison of plain least-squares and RANSAC affine warp estimation.

Both estimators are given the same correspondences: a majority that follow a known affine
transform, plus a minority of gross outliers of the kind a feature matcher produces when it
locks onto a moving object instead of the background.

Produces a four panel figure:
    1. The correspondence field, coloured by whether a point really is an outlier.
    2. A reference grid warped by the ground truth and by each estimate.
    3. Per-point residual distributions against the inlier threshold.
    4. Corner error as a function of the outlier fraction, averaged over several seeds.

Run with: uv run python tests/manual/compare_lstsq_vs_ransac.py
"""
import argparse
from typing import Tuple

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position

from motrack.cmc.components.ransac import WarpRANSACEstimator, estimate_warp_lstsq  # pylint: disable=wrong-import-position
from motrack.cmc.components.warp import apply_warp_to_points  # pylint: disable=wrong-import-position

EXTENT = 100.0
RESIDUAL_THRESHOLD = 2.0

# Rotation by 3 degrees, scale 1.02, translation (7, -4).
ANGLE = np.deg2rad(3.0)
SCALE = 1.02
GROUND_TRUTH_WARP = np.array([
    [SCALE * np.cos(ANGLE), -SCALE * np.sin(ANGLE), 7.0],
    [SCALE * np.sin(ANGLE), SCALE * np.cos(ANGLE), -4.0]
], dtype=np.float64)


def build_correspondences(
    rng: np.random.Generator,
    n_points: int,
    outlier_fraction: float,
    noise_std: float = 0.3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds correspondences following `GROUND_TRUTH_WARP`, with a fraction replaced by outliers.

    Outlier displacements are sampled as a direction plus a magnitude rather than uniformly
    from a box, so that every outlier is guaranteed to exceed the inlier threshold.

    Args:
        rng: Random number generator
        n_points: Total number of correspondences
        outlier_fraction: Fraction of correspondences that do not follow the true warp
        noise_std: Standard deviation of the localisation noise on the inliers

    Returns:
        Source points, target points, and a boolean mask of the true inliers
    """
    src = rng.uniform(0.0, EXTENT, size=(n_points, 2))
    dst = apply_warp_to_points(GROUND_TRUTH_WARP, src) + rng.normal(0.0, noise_std, size=(n_points, 2))

    n_outliers = int(round(n_points * outlier_fraction))
    true_inliers = np.ones(n_points, dtype=bool)
    if n_outliers > 0:
        true_inliers[-n_outliers:] = False
        angles = rng.uniform(0.0, 2.0 * np.pi, size=n_outliers)
        magnitudes = rng.uniform(10.0 * RESIDUAL_THRESHOLD, EXTENT, size=n_outliers)
        dst[-n_outliers:] += np.stack([magnitudes * np.cos(angles), magnitudes * np.sin(angles)], axis=1)

    return src, dst, true_inliers


def corner_error(warp: np.ndarray, reference: np.ndarray) -> float:
    """
    Measures how far a warp disagrees with a reference warp over the working area.

    Comparing matrices entry by entry mixes units, so the two warps are instead applied to
    the corners of the working area and the largest displacement is reported in pixels.

    Args:
        warp: Estimated affine 2x3 matrix
        reference: Reference affine 2x3 matrix

    Returns:
        Largest corner displacement, in pixels
    """
    corners = np.array([[0.0, 0.0], [EXTENT, 0.0], [0.0, EXTENT], [EXTENT, EXTENT]])
    return float(np.max(np.linalg.norm(apply_warp_to_points(warp, corners) - apply_warp_to_points(reference, corners), axis=1)))


def _reference_grid(n_lines: int = 6, n_samples: int = 30) -> np.ndarray:
    """
    Builds a grid of polylines covering the working area.

    Args:
        n_lines: Number of grid lines per axis
        n_samples: Number of samples per grid line

    Returns:
        Grid polylines of shape (2 * n_lines, n_samples, 2)
    """
    positions = np.linspace(0.0, EXTENT, n_lines)
    samples = np.linspace(0.0, EXTENT, n_samples)
    horizontal = np.stack([np.stack([samples, np.full_like(samples, p)], axis=1) for p in positions])
    vertical = np.stack([np.stack([np.full_like(samples, p), samples], axis=1) for p in positions])
    return np.concatenate([horizontal, vertical])


def plot_correspondences(ax: plt.Axes, src: np.ndarray, dst: np.ndarray, true_inliers: np.ndarray, max_arrows: int = 70) -> None:
    """
    Draws the correspondence field, coloured by true inlier status.

    Outliers are long and randomly oriented while inliers are short and coherent, so at full
    density the outliers cover the very structure the panel exists to show. Only a subsample
    is drawn, with the outliers faded and the inliers on top.
    """
    shown = np.zeros(len(src), dtype=bool)
    shown[np.linspace(0, len(src) - 1, min(max_arrows, len(src))).astype(int)] = True

    layers = [
        (~true_inliers & shown, 'tab:red', 0.35, 0.003, 'outlier'),
        (true_inliers & shown, 'tab:blue', 0.95, 0.005, 'inlier')
    ]
    for mask, color, alpha, width, label in layers:
        if not mask.any():
            continue
        deltas = dst[mask] - src[mask]
        ax.quiver(
            src[mask, 0], src[mask, 1], deltas[:, 0], deltas[:, 1],
            color=color, alpha=alpha, angles='xy', scale_units='xy', scale=1.0, width=width, label=label
        )

    ax.set_title(f'Correspondences ({int((~true_inliers).sum())} outliers of {len(src)}, {int(shown.sum())} drawn)')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_aspect('equal')


def plot_warped_grid(ax: plt.Axes, lstsq_warp: np.ndarray, ransac_warp: np.ndarray) -> None:
    """
    Draws a reference grid warped by the ground truth and by each estimate.
    """
    grid = _reference_grid()
    styles = [
        (GROUND_TRUTH_WARP, 'black', '-', 1.4, 'ground truth'),
        (lstsq_warp, 'tab:red', '--', 1.2, 'least-squares'),
        (ransac_warp, 'tab:green', ':', 1.6, 'RANSAC')
    ]
    for warp, color, linestyle, linewidth, label in styles:
        for i, polyline in enumerate(grid):
            warped = apply_warp_to_points(warp, polyline)
            ax.plot(
                warped[:, 0], warped[:, 1],
                color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.85,
                label=label if i == 0 else None
            )

    ax.set_title('Working area warped by each estimate')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_aspect('equal')


def plot_residuals(ax: plt.Axes, src: np.ndarray, dst: np.ndarray, lstsq_warp: np.ndarray, ransac_warp: np.ndarray) -> None:
    """
    Draws the per-point residual distribution of each estimate.
    """
    bins = np.linspace(0.0, 40.0, 60)
    for warp, color, label in [(lstsq_warp, 'tab:red', 'least-squares'), (ransac_warp, 'tab:green', 'RANSAC')]:
        residuals = np.linalg.norm(apply_warp_to_points(warp, src) - dst, axis=1)
        ax.hist(np.clip(residuals, bins[0], bins[-1]), bins=bins, color=color, alpha=0.55, label=label)

    ax.axvline(RESIDUAL_THRESHOLD, color='black', linestyle='--', linewidth=1.0, label='inlier threshold')
    ax.set_title('Per-point residuals')
    ax.set_xlabel(f'residual (px, clipped at {bins[-1]:.0f})')
    ax.set_ylabel('count')
    ax.set_yscale('log')
    ax.legend(fontsize=8)


def plot_sweep(ax: plt.Axes, n_points: int, n_seeds: int) -> None:
    """
    Draws corner error against outlier fraction, averaged over several seeds.
    """
    fractions = np.arange(0.0, 0.75, 0.05)
    lstsq_errors, ransac_errors = [], []

    for fraction in fractions:
        lstsq_run, ransac_run = [], []
        for seed in range(n_seeds):
            rng = np.random.default_rng(1000 + seed)
            src, dst, _ = build_correspondences(rng, n_points, float(fraction))

            lstsq_warp, _ = estimate_warp_lstsq(src, dst)
            estimator = WarpRANSACEstimator(
                residual_threshold=RESIDUAL_THRESHOLD, max_iterations=2000, min_inliers=10, max_skips=500, seed=seed
            )
            lstsq_run.append(corner_error(lstsq_warp, GROUND_TRUTH_WARP))
            ransac_run.append(corner_error(estimator.estimate(src, dst).warp, GROUND_TRUTH_WARP))

        lstsq_errors.append(np.mean(lstsq_run))
        ransac_errors.append(np.mean(ransac_run))

    ax.plot(fractions * 100, lstsq_errors, color='tab:red', marker='o', markersize=3, label='least-squares')
    ax.plot(fractions * 100, ransac_errors, color='tab:green', marker='s', markersize=3, label='RANSAC')
    ax.axhline(RESIDUAL_THRESHOLD, color='black', linestyle='--', linewidth=1.0, label='inlier threshold')

    ax.set_title(f'Corner error vs outlier fraction (mean of {n_seeds} seeds)')
    ax.set_xlabel('outliers (%)')
    ax.set_ylabel('max corner error (px)')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def main() -> None:
    """
    Builds one correspondence set, compares both estimators on it, and writes the figure.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-points', type=int, default=200, help='Number of correspondences')
    parser.add_argument('--outlier-fraction', type=float, default=0.4, help='Outlier fraction for the single-run panels')
    parser.add_argument('--n-seeds', type=int, default=5, help='Seeds averaged per point of the sweep')
    parser.add_argument('--seed', type=int, default=0, help='Seed for the single-run panels')
    parser.add_argument('--output', type=str, default='lstsq_vs_ransac.png', help='Output figure path')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    src, dst, true_inliers = build_correspondences(rng, args.n_points, args.outlier_fraction)

    lstsq_warp, lstsq_degenerate = estimate_warp_lstsq(src, dst)
    estimate = WarpRANSACEstimator(
        residual_threshold=RESIDUAL_THRESHOLD, max_iterations=2000, min_inliers=10, max_skips=500, seed=args.seed
    ).estimate(src, dst)

    print(f'correspondences   : {args.n_points} ({int((~true_inliers).sum())} outliers)')
    print(f'least-squares     : corner error {corner_error(lstsq_warp, GROUND_TRUTH_WARP):8.3f} px, degenerate={lstsq_degenerate}')
    print(f'RANSAC            : corner error {corner_error(estimate.warp, GROUND_TRUTH_WARP):8.3f} px, success={estimate.success}')
    print(f'RANSAC inliers    : {estimate.n_inliers}/{args.n_points} (ratio {estimate.inlier_ratio:.3f})')
    print(f'RANSAC iterations : {estimate.n_iterations} ({estimate.n_degenerate} degenerate samples)')
    print(f'true outliers kept: {int(estimate.inliers_mask[~true_inliers].sum())}')
    print(f'true inliers lost : {int((~estimate.inliers_mask[true_inliers]).sum())}')

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    plot_correspondences(axes[0, 0], src, dst, true_inliers)
    plot_warped_grid(axes[0, 1], lstsq_warp, estimate.warp)
    plot_residuals(axes[1, 0], src, dst, lstsq_warp, estimate.warp)
    plot_sweep(axes[1, 1], args.n_points, args.n_seeds)

    fig.suptitle('Affine warp estimation: least-squares vs RANSAC', fontsize=14)
    fig.tight_layout()
    fig.savefig(args.output, dpi=140)
    print(f'\nfigure written to {args.output}')


if __name__ == '__main__':
    main()
