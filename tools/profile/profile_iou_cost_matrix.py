"""
Profile IoU cost matrix formation.

Benchmarks:
1. IoUAssociation.form_cost_matrix() at various (T x D) sizes
2. Per-call PredBBox.iou() vs vectorized numpy IoU
"""
from tools.profile._common import (
    benchmark, print_table, result_cells,
    generate_bboxes, generate_tracklets, bboxes_to_xyxy_array,
    numpy_vectorized_iou,
)
from motrack.tracker.matching.algorithms.iou import IoUAssociation


def profile_cost_matrix() -> None:
    matcher = IoUAssociation(match_threshold=0.3, fuse_score=False)

    scenarios = [
        ('Small',      10,  10),
        ('Medium',     50,  50),
        ('Large',     100, 100),
        ('Asymmetric', 20,  80),
        ('Worst case',100, 200),
    ]

    headers = ['Scenario', 'T x D', 'Mean (ms)', 'Std (ms)', 'Min (ms)', 'Median (ms)']
    rows = []

    for name, n_t, n_d in scenarios:
        tracklets = generate_tracklets(n_t, seed=42)
        tracklet_bboxes = generate_bboxes(n_t, seed=42)
        detections = generate_bboxes(n_d, seed=123)

        def run():
            matcher.form_cost_matrix(tracklet_bboxes, detections, tracklets=tracklets)

        r = benchmark(run, n_warmup=3, n_runs=30)
        rows.append([name, f'{n_t} x {n_d}'] + result_cells(r))

    print_table('IoU Cost Matrix Formation', headers, rows)


def profile_iou_per_call() -> None:
    n = 100
    bboxes_a = generate_bboxes(n, seed=42)
    bboxes_b = generate_bboxes(n, seed=123)
    arr_a = bboxes_to_xyxy_array(bboxes_a)
    arr_b = bboxes_to_xyxy_array(bboxes_b)

    def run_python_iou():
        for a in bboxes_a:
            for b in bboxes_b:
                a.iou(b)

    def run_numpy_iou():
        numpy_vectorized_iou(arr_a, arr_b)

    r_python = benchmark(run_python_iou, n_warmup=2, n_runs=10)
    r_numpy = benchmark(run_numpy_iou, n_warmup=3, n_runs=50)

    n_calls = n * n
    headers = ['Method', 'Calls', 'Total (ms)', 'Per-call (us)', 'Speedup']
    rows = [
        ['PredBBox.iou()', str(n_calls), f'{r_python.mean_ms:.3f}',
         f'{r_python.mean_ms / n_calls * 1000:.3f}', '1.0x'],
        ['numpy vectorized', str(n_calls), f'{r_numpy.mean_ms:.3f}',
         f'{r_numpy.mean_ms / n_calls * 1000:.3f}',
         f'{r_python.mean_ms / max(r_numpy.mean_ms, 0.001):.1f}x'],
    ]
    print_table('Per-call IoU: Python vs Numpy (100x100)', headers, rows)

    # Correctness check
    python_iou = [[a.iou(b) for b in bboxes_b] for a in bboxes_a]
    numpy_iou = numpy_vectorized_iou(arr_a, arr_b)
    max_diff = max(abs(python_iou[i][j] - numpy_iou[i, j])
                   for i in range(n) for j in range(n))
    print(f'Correctness check: max |python - numpy| = {max_diff:.2e}')


if __name__ == '__main__':
    profile_cost_matrix()
    profile_iou_per_call()
