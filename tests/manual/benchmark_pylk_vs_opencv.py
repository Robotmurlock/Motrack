"""
Benchmarks the custom pyramidal Lucas-Kanade estimator against `cv2.calcOpticalFlowPyrLK`.

Measures two things, both on the settings the CMC pipeline will actually use:

- **Accuracy**: agreement with a known synthetic displacement, and agreement between the two
  implementations on a real MOT17 frame pair, where no ground truth exists.
- **Speed**: wall-clock per frame as a function of the number of tracked points and the
  pyramid depth.

Run with: uv run python tests/manual/benchmark_pylk_vs_opencv.py
"""
import argparse
import glob
import os
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from motrack.cmc.components.pylk import PyLucasKanadeEstimator

DEFAULT_SCENE = '/media/home/MOT17-orig/val/MOT17-13-FRCNN-H2/img1'
TARGET_LONG_EDGE = 960
WINDOW = 21
CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)


def load_frame_pair(scene_dir: str, index: int = 0) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Loads two consecutive frames resized to the pipeline's working resolution.
    """
    paths = sorted(glob.glob(os.path.join(scene_dir, '*.jpg')))
    if len(paths) < index + 2:
        return None

    frames = []
    for path in paths[index:index + 2]:
        image = cv2.imread(path)
        scale = TARGET_LONG_EDGE / max(image.shape[:2])
        image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
        frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    return frames[0], frames[1]


def synthetic_pair(shift: Tuple[float, float], height: int = 540, width: int = 960) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds a textured frame pair with an exactly known displacement.

    Both frames are cropped from one larger image so that no synthetic border is introduced.
    """
    rng = np.random.default_rng(0)
    margin = 40
    source = rng.integers(0, 256, size=(height + 2 * margin, width + 2 * margin)).astype(np.float32)
    source = cv2.normalize(cv2.GaussianBlur(source, (0, 0), sigmaX=2.0), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    dx, dy = int(shift[0]), int(shift[1])
    first = source[margin:margin + height, margin:margin + width]
    second = source[margin + dy:margin + dy + height, margin + dx:margin + dx + width]
    return np.repeat(first[:, :, None], 3, axis=2), np.repeat(second[:, :, None], 3, axis=2)


def run_opencv(prev_frame: np.ndarray, next_frame: np.ndarray, points: np.ndarray, max_level: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs the OpenCV reference implementation.

    Returns:
        Flows and a boolean mask of untracked points, matching the custom estimator's contract
    """
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(
        cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY), cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY),
        points.reshape(-1, 1, 2), None, winSize=(WINDOW, WINDOW), maxLevel=max_level, criteria=CRITERIA
    )
    return tracked.reshape(-1, 2) - points, status.ravel() == 0


def detect_points(frame: np.ndarray, count: int) -> np.ndarray:
    """
    Selects Shi-Tomasi corners, the detector the CMC pipeline pairs with this tracker.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=count, qualityLevel=0.005, minDistance=7)
    return corners.reshape(-1, 2).astype(np.float32)


def time_call(fn, repeats: int) -> float:
    """
    Returns the best wall-clock time in milliseconds over `repeats` runs.

    The minimum is reported rather than the mean because it is the least contaminated by
    scheduling noise.
    """
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        timings.append((time.perf_counter() - start) * 1000.0)
    return min(timings)


def benchmark_accuracy(max_level: int) -> List[str]:
    """
    Compares both implementations against known synthetic displacements.
    """
    lines = ['| true shift | custom error (px) | OpenCV error (px) |', '|---|---|---|']
    estimator = PyLucasKanadeEstimator(max_level=max_level)

    for shift in [(2, 0), (5, -3), (10, 0), (16, -8), (24, 0)]:
        prev_frame, next_frame = synthetic_pair(shift)
        points = detect_points(prev_frame, 200)

        ours, skips = estimator.estimate(prev_frame, next_frame, points)
        theirs, their_skips = run_opencv(prev_frame, next_frame, points, max_level - 1)

        expected = np.array([-shift[0], -shift[1]], dtype=np.float64)
        our_median = np.median(ours[~skips], axis=0)
        their_median = np.median(theirs[~their_skips], axis=0)
        lines.append(
            f'| ({shift[0]}, {shift[1]}) | {np.hypot(*(our_median - expected)):.3f} '
            f'| {np.hypot(*(their_median - expected)):.3f} |'
        )

    return lines


def benchmark_agreement(scene_dir: str, max_level: int) -> List[str]:
    """
    Compares the two implementations on real frames, where no ground truth exists.
    """
    pair = load_frame_pair(scene_dir)
    if pair is None:
        return ['_Scene not available, skipped._']

    prev_frame, next_frame = pair
    points = detect_points(prev_frame, 300)
    ours, skips = PyLucasKanadeEstimator(max_level=max_level).estimate(prev_frame, next_frame, points)
    theirs, their_skips = run_opencv(prev_frame, next_frame, points, max_level - 1)

    both = ~skips & ~their_skips
    difference = np.linalg.norm(ours[both] - theirs[both], axis=1)
    return [
        f'- Points tracked by both: **{int(both.sum())} / {len(points)}**',
        f'- Median disagreement: **{np.median(difference):.3f} px**',
        f'- 90th percentile: **{np.percentile(difference, 90):.3f} px**',
        f'- Custom rejected {int(skips.sum())}, OpenCV rejected {int(their_skips.sum())}'
    ]


def benchmark_speed(scene_dir: str, repeats: int) -> List[str]:
    """
    Measures wall-clock cost against point count and pyramid depth.
    """
    pair = load_frame_pair(scene_dir)
    prev_frame, next_frame = pair if pair is not None else synthetic_pair((5, -3))

    lines = [
        '| points | levels | custom (ms) | OpenCV (ms) | ratio |',
        '|---|---|---|---|---|'
    ]
    for count in [50, 100, 200, 400]:
        for max_level in [3, 4]:
            points = detect_points(prev_frame, count)
            estimator = PyLucasKanadeEstimator(max_level=max_level)
            ours = time_call(lambda: estimator.estimate(prev_frame, next_frame, points), repeats)  # pylint: disable=cell-var-from-loop
            theirs = time_call(lambda: run_opencv(prev_frame, next_frame, points, max_level - 1), repeats)  # pylint: disable=cell-var-from-loop
            lines.append(f'| {len(points)} | {max_level} | {ours:.1f} | {theirs:.2f} | {ours / theirs:.0f}x |')

    return lines


def main() -> None:
    """
    Runs every benchmark and prints a markdown report to stdout.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-dir', type=str, default=DEFAULT_SCENE)
    parser.add_argument('--max-level', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()

    print(f'OpenCV {cv2.__version__}, window {WINDOW}x{WINDOW}, working resolution {TARGET_LONG_EDGE} long edge')
    print()
    print('### Accuracy against known displacement')
    print('\n'.join(benchmark_accuracy(args.max_level)))
    print()
    print('### Agreement on a real frame pair')
    print('\n'.join(benchmark_agreement(args.scene_dir, args.max_level)))
    print()
    print('### Speed')
    print('\n'.join(benchmark_speed(args.scene_dir, args.repeats)))


if __name__ == '__main__':
    main()
