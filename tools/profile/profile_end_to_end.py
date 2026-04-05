"""
Profile end-to-end tracking frame step.

Simulates the core tracking loop (predict -> cost matrix -> assign -> update -> lost)
without full ByteTracker instantiation, measuring each component's contribution.
"""
import time
import copy

import numpy as np

from tools.profile._common import (
    print_table, generate_bboxes, generate_tracklets, generate_kalman_states,
)
from motrack.library.cv.bbox import BBox, PredBBox
from motrack.library.kalman_filter.botsort_kf import BotSortKalmanFilter
from motrack.tracker.matching.algorithms.iou import IoUAssociation
from motrack.tracker.matching.utils import hungarian


N_FRAMES = 200


def simulate_frame(
    kf: BotSortKalmanFilter,
    matcher: IoUAssociation,
    filter_states: dict,
    tracklet_bboxes: list,
    tracklets: list,
    detections: list,
) -> dict:
    """Simulate one frame and return timing breakdown in seconds."""
    timings = {}

    # 1. Batch predict all tracklets
    t0 = time.perf_counter()
    states = [(filter_states[t.id][0], filter_states[t.id][1]) for t in tracklets]
    means = np.stack([s[0] for s in states])
    covs = np.stack([s[1] for s in states])
    means_hat, covs_hat = kf.multi_predict(means, covs)

    prior_bboxes = []
    for i, (t, bbox) in enumerate(zip(tracklets, tracklet_bboxes)):
        filter_states[t.id] = (means_hat[i], covs_hat[i])
        mean_proj, _ = kf.project(means_hat[i], covs_hat[i])
        prior_bbox = PredBBox.create(
            bbox=BBox.from_xywh(*mean_proj.tolist(), clip=False),
            label=bbox.label, conf=bbox.conf
        )
        prior_bboxes.append(prior_bbox)
    timings['predict_all'] = time.perf_counter() - t0

    # 2. Form cost matrix
    t0 = time.perf_counter()
    cost_matrix = matcher.form_cost_matrix(prior_bboxes, detections, tracklets=tracklets)
    timings['form_cost_matrix'] = time.perf_counter() - t0

    # 3. Hungarian assignment
    t0 = time.perf_counter()
    matches, unmatched_t, unmatched_d = hungarian(cost_matrix)
    timings['hungarian'] = time.perf_counter() - t0

    # 4. Update matched tracklets
    t0 = time.perf_counter()
    for t_i, d_i in matches:
        mean_hat, cov_hat = filter_states[tracklets[t_i].id]
        measurement = detections[d_i].as_numpy_xywh()
        mean_upd, cov_upd = kf.update(mean_hat, cov_hat, measurement)
        filter_states[tracklets[t_i].id] = (mean_upd, cov_upd)
    timings['update_matched'] = time.perf_counter() - t0

    # 5. Handle lost (missing) tracklets
    t0 = time.perf_counter()
    for t_i in unmatched_t:
        # missing() is a no-op for bot-sort (returns prior), but we call predict again
        # to mirror the real pipeline's filter.missing() cost
        _ = filter_states[tracklets[t_i].id]
    timings['handle_lost'] = time.perf_counter() - t0

    return timings


def profile_scenario(n_tracklets: int, n_detections: int) -> None:
    kf = BotSortKalmanFilter()
    matcher = IoUAssociation(match_threshold=0.3, fuse_score=False)

    tracklets = generate_tracklets(n_tracklets, seed=42)
    tracklet_bboxes = generate_bboxes(n_tracklets, seed=42)
    detections = generate_bboxes(n_detections, seed=123)
    states = generate_kalman_states(n_tracklets, seed=42)
    filter_states = {t.id: states[i] for i, t in enumerate(tracklets)}

    # Warmup
    for _ in range(3):
        fs_copy = {k: (v[0].copy(), v[1].copy()) for k, v in filter_states.items()}
        simulate_frame(kf, matcher, fs_copy, tracklet_bboxes, tracklets, detections)

    # Benchmark
    all_timings = {k: [] for k in ['predict_all', 'form_cost_matrix', 'hungarian', 'update_matched', 'handle_lost']}
    for _ in range(N_FRAMES):
        fs_copy = {k: (v[0].copy(), v[1].copy()) for k, v in filter_states.items()}
        timings = simulate_frame(kf, matcher, fs_copy, tracklet_bboxes, tracklets, detections)
        for k, v in timings.items():
            all_timings[k].append(v)

    headers = ['Component', 'Mean (ms)', '% of Total']
    rows = []
    means = {k: np.mean(v) * 1000 for k, v in all_timings.items()}
    total = sum(means.values())

    for k in ['predict_all', 'form_cost_matrix', 'hungarian', 'update_matched', 'handle_lost']:
        pct = means[k] / max(total, 0.001) * 100
        rows.append([k, f'{means[k]:.3f}', f'{pct:.1f}%'])
    rows.append(['TOTAL', f'{total:.3f}', '100.0%'])

    print_table(f'End-to-End Frame Step (T={n_tracklets}, D={n_detections})', headers, rows)


if __name__ == '__main__':
    scenarios = [
        (20, 30),
        (50, 50),
        (100, 100),
    ]
    for n_t, n_d in scenarios:
        profile_scenario(n_t, n_d)
