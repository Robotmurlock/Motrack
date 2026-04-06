"""
Shared utilities for profiling scripts: timing, synthetic data generation, output formatting.
"""
import os
import time
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from motrack.library.cv.bbox import BBox, PredBBox
from motrack.library.kalman_filter.botsort_kf import BotSortKalmanFilter
from motrack.tracker.tracklet import Tracklet, TrackletState


@dataclass
class BenchmarkResult:
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    median_ms: float
    n_runs: int


def benchmark(fn: Callable, n_warmup: int = 3, n_runs: int = 50) -> BenchmarkResult:
    """Run fn() with warmup, then time n_runs iterations."""
    for _ in range(n_warmup):
        fn()

    times_s: List[float] = []
    for _ in range(n_runs):
        start = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - start
        times_s.append(elapsed)

    times_ms = np.array(times_s) * 1000.0
    return BenchmarkResult(
        mean_ms=float(np.mean(times_ms)),
        std_ms=float(np.std(times_ms)),
        min_ms=float(np.min(times_ms)),
        max_ms=float(np.max(times_ms)),
        median_ms=float(np.median(times_ms)),
        n_runs=n_runs,
    )


def print_table(title: str, headers: List[str], rows: List[List[str]]) -> None:
    """Print a formatted ASCII table."""
    print(f'\n=== {title} ===\n')
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def fmt_row(cells: List[str]) -> str:
        parts = [cell.ljust(col_widths[i]) for i, cell in enumerate(cells)]
        return '| ' + ' | '.join(parts) + ' |'

    separator = '|-' + '-|-'.join('-' * w for w in col_widths) + '-|'
    print(fmt_row(headers))
    print(separator)
    for row in rows:
        print(fmt_row(row))
    print()


def result_cells(r: BenchmarkResult) -> List[str]:
    """Format a BenchmarkResult into table cells: [mean, std, min, median]."""
    return [f'{r.mean_ms:.3f}', f'{r.std_ms:.3f}', f'{r.min_ms:.3f}', f'{r.median_ms:.3f}']


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def generate_bboxes(n: int, seed: int = 42) -> List[PredBBox]:
    """Generate n random PredBBox objects with realistic sizes and sufficient overlap."""
    rng = np.random.RandomState(seed)
    bboxes: List[PredBBox] = []
    for _ in range(n):
        cx, cy = rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)
        w = rng.uniform(0.02, 0.15)
        h = rng.uniform(0.02, 0.15)
        x1 = max(0.0, cx - w / 2)
        y1 = max(0.0, cy - h / 2)
        x2 = min(1.0, cx + w / 2)
        y2 = min(1.0, cy + h / 2)
        label = int(rng.randint(0, 3))
        conf = float(rng.uniform(0.3, 1.0))
        bbox = PredBBox.create(BBox.from_xyxy(x1, y1, x2, y2), label=label, conf=conf)
        bboxes.append(bbox)
    return bboxes


def bboxes_to_xyxy_array(bboxes: List[PredBBox]) -> np.ndarray:
    """Convert list of PredBBox to (N, 4) xyxy numpy array."""
    return np.array([[b.upper_left.x, b.upper_left.y, b.bottom_right.x, b.bottom_right.y]
                     for b in bboxes], dtype=np.float32)


def generate_tracklets(n: int, seed: int = 42) -> List[Tracklet]:
    """Generate n Tracklet objects with ACTIVE state."""
    bboxes = generate_bboxes(n, seed=seed)
    tracklets: List[Tracklet] = []
    for i, bbox in enumerate(bboxes):
        t = Tracklet(bbox=copy.deepcopy(bbox), frame_index=10, _id=i, state=TrackletState.ACTIVE)
        tracklets.append(t)
    return tracklets


def generate_kalman_states(
    n: int, seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate n realistic (mean, covariance) Kalman filter state pairs."""
    rng = np.random.RandomState(seed)
    kf = BotSortKalmanFilter()
    states: List[Tuple[np.ndarray, np.ndarray]] = []
    for _ in range(n):
        measurement = np.array([
            rng.uniform(0.2, 0.8),
            rng.uniform(0.2, 0.8),
            rng.uniform(0.02, 0.15),
            rng.uniform(0.02, 0.15),
        ], dtype=np.float64)
        mean, cov = kf.initiate(measurement)
        # Run a few predict steps to make the state realistic
        for _ in range(rng.randint(1, 5)):
            mean, cov = kf.predict(mean, cov)
        states.append((mean, cov))
    return states


def generate_cost_matrix(
    n_tracklets: int, n_detections: int, sparsity: float = 0.3, seed: int = 42
) -> np.ndarray:
    """Generate a random cost matrix with np.inf for gated entries."""
    rng = np.random.RandomState(seed)
    cost_matrix = -rng.uniform(0.0, 1.0, size=(n_tracklets, n_detections)).astype(np.float32)
    mask = rng.random(size=(n_tracklets, n_detections)) < sparsity
    cost_matrix[mask] = np.inf
    return cost_matrix


def generate_npy_cache(
    tmp_dir: str, n_frames: int, n_detections: int = 50, seed: int = 42
) -> str:
    """Create a temporary OD cache directory with npy files. Returns cache root path."""
    rng = np.random.RandomState(seed)
    scene_name = 'scene_001'
    cache_root = os.path.join(tmp_dir, 'cache')

    for frame_idx in range(n_frames):
        frame_dir = os.path.join(cache_root, scene_name, f'{frame_idx:06d}')
        Path(frame_dir).mkdir(parents=True, exist_ok=True)
        bboxes = rng.uniform(0, 1, size=(n_detections, 4)).astype(np.float32)
        bboxes[:, 2:] += bboxes[:, :2]  # ensure x2 > x1, y2 > y1
        classes = rng.randint(0, 5, size=(n_detections,)).astype(np.float32)
        confidences = rng.uniform(0.3, 1.0, size=(n_detections,)).astype(np.float32)
        np.save(os.path.join(frame_dir, 'bboxes.npy'), bboxes)
        np.save(os.path.join(frame_dir, 'classes.npy'), classes)
        np.save(os.path.join(frame_dir, 'confidences.npy'), confidences)

    return cache_root, scene_name


def generate_npz_cache(
    tmp_dir: str, n_frames: int, n_detections: int = 50, seed: int = 42
) -> str:
    """Create a temporary OD cache with single npz files per frame. Returns cache root path."""
    rng = np.random.RandomState(seed)
    scene_name = 'scene_001'
    cache_root = os.path.join(tmp_dir, 'cache_npz')

    for frame_idx in range(n_frames):
        frame_dir = os.path.join(cache_root, scene_name, f'{frame_idx:06d}')
        Path(frame_dir).mkdir(parents=True, exist_ok=True)
        bboxes = rng.uniform(0, 1, size=(n_detections, 4)).astype(np.float32)
        bboxes[:, 2:] += bboxes[:, :2]
        classes = rng.randint(0, 5, size=(n_detections,)).astype(np.float32)
        confidences = rng.uniform(0.3, 1.0, size=(n_detections,)).astype(np.float32)
        np.savez(os.path.join(frame_dir, 'data.npz'),
                 bboxes=bboxes, classes=classes, confidences=confidences)

    return cache_root, scene_name


# ---------------------------------------------------------------------------
# Vectorized IoU reference implementation
# ---------------------------------------------------------------------------

def numpy_vectorized_iou(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
    """
    Compute IoU matrix between two sets of bboxes using vectorized numpy.

    Args:
        bboxes_a: (N, 4) xyxy array
        bboxes_b: (M, 4) xyxy array

    Returns:
        (N, M) IoU matrix
    """
    x1 = np.maximum(bboxes_a[:, 0:1], bboxes_b[:, 0:1].T)
    y1 = np.maximum(bboxes_a[:, 1:2], bboxes_b[:, 1:2].T)
    x2 = np.minimum(bboxes_a[:, 2:3], bboxes_b[:, 2:3].T)
    y2 = np.minimum(bboxes_a[:, 3:4], bboxes_b[:, 3:4].T)

    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    area_a = (bboxes_a[:, 2] - bboxes_a[:, 0]) * (bboxes_a[:, 3] - bboxes_a[:, 1])
    area_b = (bboxes_b[:, 2] - bboxes_b[:, 0]) * (bboxes_b[:, 3] - bboxes_b[:, 1])

    union = area_a[:, None] + area_b[None, :] - intersection
    union = np.maximum(union, 1e-8)

    return intersection / union
