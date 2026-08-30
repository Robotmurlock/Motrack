"""
Manual demonstration of the custom pyramidal Lucas-Kanade optical flow estimator.

Produces a four panel figure:
    1. Real camera motion. Shi-Tomasi features on a consecutive MOT17 frame pair, tracked
       with PyLK, with the resulting correspondences fed to the RANSAC affine estimator.
       Arrows are coloured by whether RANSAC accepted them, so the panel shows both stages
       of the correspondence pipeline at once.
    2. The pyramid itself, one image per level.
    3. Convergence: the size of the correction at each iteration, per level.
    4. What pyramid depth buys: accuracy against displacement magnitude for each `max_level`.

Run with: uv run python tests/manual/visualize_pylk.py
"""
import argparse
import glob
import os
from typing import Optional, Tuple

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
import cv2  # pylint: disable=wrong-import-position

from motrack.cmc.components.pylk import PyLucasKanadeEstimator, min_eigenvalue2x2  # pylint: disable=wrong-import-position
from motrack.cmc.components.ransac import WarpRANSACEstimator  # pylint: disable=wrong-import-position

DEFAULT_SCENE = '/media/home/MOT17-orig/val/MOT17-13-FRCNN-H2/img1'
TARGET_LONG_EDGE = 960


def load_frame_pair(scene_dir: str, index: int, target_long_edge: int = TARGET_LONG_EDGE) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Loads two consecutive frames, resized the way the CMC pipeline will resize them.

    Args:
        scene_dir: Directory holding the scene images
        index: Index of the first frame
        target_long_edge: Target size of the longer image side

    Returns:
        Consecutive RGB frames, or None when the scene is unavailable
    """
    paths = sorted(glob.glob(os.path.join(scene_dir, '*.jpg')))
    if len(paths) < index + 2:
        return None

    frames = []
    for path in paths[index:index + 2]:
        image = cv2.imread(path)
        scale = target_long_edge / max(image.shape[:2])
        image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
        frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    return frames[0], frames[1]


def synthetic_frame(height: int = 360, width: int = 480, seed: int = 0) -> np.ndarray:
    """
    Builds a deterministic textured frame, used when the dataset is unavailable and for the
    controlled panels.
    """
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, size=(height, width)).astype(np.float32)
    blurred = cv2.GaussianBlur(noise, (0, 0), sigmaX=2.0)
    gray = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def plot_real_motion(ax: plt.Axes, prev_frame: np.ndarray, next_frame: np.ndarray, max_features: int) -> str:
    """
    Tracks Shi-Tomasi features across a real frame pair and separates the correspondences
    into RANSAC inliers and outliers.

    Returns:
        A short summary line for the console
    """
    gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
    points = cv2.goodFeaturesToTrack(gray, maxCorners=max_features, qualityLevel=0.01, minDistance=12)
    points = points.reshape(-1, 2).astype(np.float32)

    flows, skips = PyLucasKanadeEstimator(max_level=4).estimate(prev_frame, next_frame, points)
    tracked = ~skips

    estimate = WarpRANSACEstimator(residual_threshold=1.5, max_iterations=800, min_inliers=10, max_skips=200).estimate(
        points[tracked], points[tracked] + flows[tracked]
    )
    inliers = np.zeros(len(points), dtype=bool)
    inliers[np.flatnonzero(tracked)[estimate.inliers_mask]] = True

    ax.imshow(prev_frame)
    layers = [
        (skips, 'tab:gray', 0.45, 'untrackable'),
        (tracked & ~inliers, 'tab:red', 0.95, 'RANSAC outlier'),
        (inliers, 'tab:green', 0.95, 'RANSAC inlier (background)')
    ]
    for mask, color, alpha, label in layers:
        if not mask.any():
            continue
        ax.quiver(
            points[mask, 0], points[mask, 1], flows[mask, 0], -flows[mask, 1],
            color=color, alpha=alpha, angles='xy', scale_units='xy', scale=0.06, width=0.003, label=label
        )

    ax.set_title(f'Real camera motion: PyLK flow, coloured by RANSAC verdict\n(arrows scaled x{1/0.06:.0f})')
    ax.legend(loc='lower right', fontsize=7)
    ax.set_axis_off()

    return (f'   tracked {int(tracked.sum())}/{len(points)}, RANSAC inliers {estimate.n_inliers} '
            f'(ratio {estimate.inlier_ratio:.2f}), success={estimate.success}')


def plot_pyramid(ax: plt.Axes, frame: np.ndarray, max_level: int) -> None:
    """
    Draws each pyramid level side by side, at its true relative size.
    """
    estimator = PyLucasKanadeEstimator(max_level=max_level)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    levels = [estimator._scale_frame_to_level(gray, level) for level in range(max_level)]  # pylint: disable=protected-access
    label_band = 18
    canvas = np.full(
        (levels[0].shape[0] + label_band, sum(level.shape[1] for level in levels) + 10 * len(levels)),
        255, dtype=np.uint8
    )

    offset = 0
    for level, image in enumerate(levels):
        canvas[:image.shape[0], offset:offset + image.shape[1]] = image
        ax.text(
            offset + 2, levels[0].shape[0] + label_band - 4,
            f'L{level} {image.shape[1]}x{image.shape[0]}', fontsize=7, color='black'
        )
        offset += image.shape[1] + 10

    ax.imshow(canvas, cmap='gray', vmin=0, vmax=255)
    ax.set_title('Pyramid levels (window stays 21px, so its footprint doubles per level)')
    ax.set_axis_off()


def plot_convergence(ax: plt.Axes, max_level: int, shift: Tuple[float, float]) -> None:
    """
    Draws the magnitude of the correction at each iteration, for every pyramid level.
    """
    frame = synthetic_frame()
    shifted = cv2.warpAffine(
        frame, np.array([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]], dtype=np.float32),
        (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )
    estimator = PyLucasKanadeEstimator(max_level=max_level)
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    next_gray = cv2.cvtColor(shifted, cv2.COLOR_RGB2GRAY)
    point = np.array([frame.shape[1] / 2, frame.shape[0] / 2], dtype=np.float32)

    for level in range(max_level - 1, -1, -1):
        scale = 2 ** level
        prev_level = estimator._scale_frame_to_level(prev_gray, level)  # pylint: disable=protected-access
        next_level = estimator._scale_frame_to_level(next_gray, level)  # pylint: disable=protected-access
        patch_p = estimator._sample_patch(prev_level, point, level)  # pylint: disable=protected-access
        grad_x, grad_y = estimator._compute_gradients(patch_p)  # pylint: disable=protected-access
        gradient_matrix = np.array([
            [np.sum(grad_x * grad_x), np.sum(grad_x * grad_y)],
            [np.sum(grad_x * grad_y), np.sum(grad_y * grad_y)]
        ])
        if min_eigenvalue2x2(gradient_matrix, patch_p.size) < 1e-3:
            continue

        flow = np.zeros(2, dtype=np.float32)
        steps = []
        for _ in range(20):
            patch_q = estimator._sample_patch(next_level, point + flow * scale, level)  # pylint: disable=protected-access
            residual = patch_q.astype(np.float32) - patch_p.astype(np.float32)
            rhs = -np.array([np.sum(grad_x * residual), np.sum(grad_y * residual)])
            delta = np.linalg.solve(gradient_matrix, rhs)
            flow = flow + delta
            steps.append(np.linalg.norm(delta))

        ax.plot(range(1, len(steps) + 1), steps, marker='o', markersize=3, label=f'level {level} (scale {scale})')

    ax.axhline(0.1, color='black', linestyle='--', linewidth=1.0, label='convergence threshold')
    ax.set_title(f'Correction size per iteration, shift {shift}\n(steps grow before they collapse)')
    ax.set_xlabel('iteration')
    ax.set_ylabel('|d| (level px)')
    ax.set_yscale('log')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)


def plot_depth_vs_displacement(ax: plt.Axes) -> None:
    """
    Draws recovery error against displacement magnitude for several pyramid depths.

    Both frames are cropped from one larger image rather than warped, so no synthetic border
    is introduced at the shifts where a coarse window would otherwise reach it.
    """
    source = cv2.cvtColor(synthetic_frame(height=440, width=620, seed=1), cv2.COLOR_RGB2GRAY)
    height, width = 360, 480
    xs = np.arange(90, width - 90, 45, dtype=np.float32)
    ys = np.arange(90, height - 90, 45, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float32)

    shifts = np.arange(0, 26, 2)
    for max_level in [1, 2, 3, 4]:
        errors = []
        estimator = PyLucasKanadeEstimator(max_level=max_level)
        for shift in shifts:
            prev_frame = np.repeat(source[40:40 + height, 40:40 + width][:, :, None], 3, axis=2)
            next_frame = np.repeat(source[40:40 + height, 40 + shift:40 + shift + width][:, :, None], 3, axis=2)
            flows, skips = estimator.estimate(prev_frame, next_frame, points)
            valid = flows[~skips]
            median = np.median(valid, axis=0) if len(valid) else np.array([np.nan, np.nan])
            errors.append(np.hypot(median[0] + shift, median[1]))
        ax.plot(shifts, errors, marker='o', markersize=3, label=f'max_level={max_level}')

    ax.axhline(1.0, color='black', linestyle='--', linewidth=1.0, label='1 px error')
    ax.set_title('Each level roughly doubles the trackable displacement')
    ax.set_xlabel('true displacement (px)')
    ax.set_ylabel('recovery error (px)')
    ax.set_yscale('log')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)


def main() -> None:
    """
    Builds the figure and writes it to disk.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-dir', type=str, default=DEFAULT_SCENE, help='MOT17 image directory for panel 1')
    parser.add_argument('--frame-index', type=int, default=0, help='Index of the first frame of the pair')
    parser.add_argument('--max-features', type=int, default=150, help='Shi-Tomasi corners tracked in panel 1')
    parser.add_argument('--output', type=str, default='pylk.png', help='Output figure path')
    args = parser.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    pair = load_frame_pair(args.scene_dir, args.frame_index)
    if pair is None:
        print(f'scene not found at {args.scene_dir}, falling back to a synthetic pair for panel 1')
        prev_frame = synthetic_frame()
        next_frame = cv2.warpAffine(
            prev_frame, np.array([[1.0, 0.0, 6.0], [0.0, 1.0, -3.0]], dtype=np.float32),
            (prev_frame.shape[1], prev_frame.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )
    else:
        prev_frame, next_frame = pair

    print('panel 1: tracking a real frame pair')
    print(plot_real_motion(axes[0, 0], prev_frame, next_frame, args.max_features))
    plot_pyramid(axes[0, 1], prev_frame, max_level=4)
    plot_convergence(axes[1, 0], max_level=3, shift=(10.0, 0.0))
    plot_depth_vs_displacement(axes[1, 1])

    fig.suptitle('Pyramidal Lucas-Kanade optical flow', fontsize=14)
    fig.tight_layout()
    fig.savefig(args.output, dpi=140)
    print(f'\nfigure written to {args.output}')


if __name__ == '__main__':
    main()
