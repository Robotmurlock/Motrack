"""
Profile OD detection cache loading.

Benchmarks:
1. Current approach: 3 separate np.load calls per frame
2. Single npz file per frame
3. Memory-mapped loading
4. Sequential scene scan (many frames)
"""
import os
import shutil
import tempfile

import numpy as np

from tools.profile._common import (
    benchmark, print_table, generate_npy_cache, generate_npz_cache,
)


def load_3_npy(frame_dir: str):
    """Current approach: 3 separate loads."""
    bboxes = np.load(os.path.join(frame_dir, 'bboxes.npy'))
    classes = np.load(os.path.join(frame_dir, 'classes.npy'))
    confidences = np.load(os.path.join(frame_dir, 'confidences.npy'))
    return bboxes, classes, confidences


def load_1_npz(frame_dir: str):
    """Single npz load."""
    data = np.load(os.path.join(frame_dir, 'data.npz'))
    return data['bboxes'], data['classes'], data['confidences']


def load_3_npy_mmap(frame_dir: str):
    """Memory-mapped loading."""
    bboxes = np.load(os.path.join(frame_dir, 'bboxes.npy'), mmap_mode='r')
    classes = np.load(os.path.join(frame_dir, 'classes.npy'), mmap_mode='r')
    confidences = np.load(os.path.join(frame_dir, 'confidences.npy'), mmap_mode='r')
    return bboxes, classes, confidences


def profile_per_frame() -> None:
    scenarios = [
        ('Small',  10),
        ('Medium', 50),
        ('Large', 200),
    ]

    headers = ['Scenario', 'N dets', '3x npy (ms)', '1x npz (ms)', 'mmap (ms)']
    rows = []

    for name, n_dets in scenarios:
        tmp_dir = tempfile.mkdtemp()
        try:
            cache_root_npy, scene = generate_npy_cache(tmp_dir, n_frames=10, n_detections=n_dets)
            cache_root_npz, _ = generate_npz_cache(tmp_dir, n_frames=10, n_detections=n_dets)

            frame_dir_npy = os.path.join(cache_root_npy, scene, f'{0:06d}')
            frame_dir_npz = os.path.join(cache_root_npz, scene, f'{0:06d}')

            r_npy = benchmark(lambda: load_3_npy(frame_dir_npy), n_warmup=5, n_runs=100)
            r_npz = benchmark(lambda: load_1_npz(frame_dir_npz), n_warmup=5, n_runs=100)
            r_mmap = benchmark(lambda: load_3_npy_mmap(frame_dir_npy), n_warmup=5, n_runs=100)

            rows.append([name, str(n_dets),
                         f'{r_npy.mean_ms:.3f}', f'{r_npz.mean_ms:.3f}', f'{r_mmap.mean_ms:.3f}'])
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    print_table('OD Cache Per-Frame Loading', headers, rows)


def profile_sequential_scan() -> None:
    n_frames = 1000
    n_dets = 50
    tmp_dir = tempfile.mkdtemp()

    try:
        cache_root_npy, scene = generate_npy_cache(tmp_dir, n_frames=n_frames, n_detections=n_dets)
        cache_root_npz, _ = generate_npz_cache(tmp_dir, n_frames=n_frames, n_detections=n_dets)

        def scan_npy():
            for i in range(n_frames):
                frame_dir = os.path.join(cache_root_npy, scene, f'{i:06d}')
                load_3_npy(frame_dir)

        def scan_npz():
            for i in range(n_frames):
                frame_dir = os.path.join(cache_root_npz, scene, f'{i:06d}')
                load_1_npz(frame_dir)

        def scan_mmap():
            for i in range(n_frames):
                frame_dir = os.path.join(cache_root_npy, scene, f'{i:06d}')
                load_3_npy_mmap(frame_dir)

        r_npy = benchmark(scan_npy, n_warmup=1, n_runs=5)
        r_npz = benchmark(scan_npz, n_warmup=1, n_runs=5)
        r_mmap = benchmark(scan_mmap, n_warmup=1, n_runs=5)

        headers = ['Strategy', f'Total {n_frames} frames (ms)', 'Per-frame (ms)']
        rows = [
            ['3x npy (current)', f'{r_npy.mean_ms:.1f}', f'{r_npy.mean_ms / n_frames:.3f}'],
            ['1x npz', f'{r_npz.mean_ms:.1f}', f'{r_npz.mean_ms / n_frames:.3f}'],
            ['3x npy mmap', f'{r_mmap.mean_ms:.1f}', f'{r_mmap.mean_ms / n_frames:.3f}'],
        ]
        print_table(f'Sequential Scene Scan ({n_frames} frames, {n_dets} dets/frame)', headers, rows)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    profile_per_frame()
    profile_sequential_scan()
