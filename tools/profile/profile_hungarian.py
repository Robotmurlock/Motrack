"""
Profile linear assignment solvers.

Benchmarks:
1. Current hungarian() with augmentation
2. Direct scipy.optimize.linear_sum_assignment on rectangular matrix
3. Greedy assignment
4. Augmentation cost in isolation
"""
import numpy as np
import scipy.optimize

from tools.profile._common import benchmark, print_table, result_cells, generate_cost_matrix
from motrack.tracker.matching.utils import hungarian, greedy, INF


def scipy_direct(cost_matrix: np.ndarray):
    """Direct scipy assignment without augmentation."""
    n_tracklets, n_detections = cost_matrix.shape
    # Replace inf with a large finite value for scipy
    finite_matrix = cost_matrix.copy()
    finite_matrix[finite_matrix == np.inf] = INF

    row_indices, col_indices = scipy.optimize.linear_sum_assignment(finite_matrix)

    matches = []
    matched_tracklets = set()
    matched_detections = set()
    for r_i, c_i in zip(row_indices, col_indices):
        if cost_matrix[r_i, c_i] == np.inf:
            continue
        matches.append((r_i, c_i))
        matched_tracklets.add(r_i)
        matched_detections.add(c_i)

    unmatched_tracklets = list(set(range(n_tracklets)) - matched_tracklets)
    unmatched_detections = list(set(range(n_detections)) - matched_detections)
    return matches, unmatched_tracklets, unmatched_detections


def profile_assignment() -> None:
    scenarios = [
        ('Small',     10,  10),
        ('Medium',    50,  50),
        ('Large',    100, 100),
        ('Asym-wide', 20,  80),
        ('Asym-tall', 80,  20),
    ]

    headers = ['Scenario', 'Size', 'Hungarian+Aug (ms)', 'Scipy Direct (ms)',
               'Greedy (ms)', 'Speedup (direct/aug)']
    rows = []

    for name, n_t, n_d in scenarios:
        cost_matrix = generate_cost_matrix(n_t, n_d, sparsity=0.3, seed=42)

        r_hung = benchmark(lambda: hungarian(cost_matrix), n_warmup=3, n_runs=50)
        r_direct = benchmark(lambda: scipy_direct(cost_matrix), n_warmup=3, n_runs=50)
        r_greedy = benchmark(lambda: greedy(cost_matrix), n_warmup=3, n_runs=50)

        speedup = r_hung.mean_ms / max(r_direct.mean_ms, 0.001)
        rows.append([
            name, f'{n_t}x{n_d}',
            f'{r_hung.mean_ms:.3f}', f'{r_direct.mean_ms:.3f}',
            f'{r_greedy.mean_ms:.3f}', f'{speedup:.2f}x',
        ])

    print_table('Linear Assignment Solvers', headers, rows)


def profile_augmentation_overhead() -> None:
    scenarios = [
        ('Small',     10,  10),
        ('Medium',    50,  50),
        ('Large',    100, 100),
    ]

    headers = ['Scenario', 'Size', 'Augmentation only (ms)', 'Full hungarian (ms)', 'Aug % of total']
    rows = []

    for name, n_t, n_d in scenarios:
        cost_matrix = generate_cost_matrix(n_t, n_d, sparsity=0.3, seed=42)

        def augment_only():
            augmentation = INF * np.ones(shape=(n_t, n_t), dtype=np.float32) - np.eye(n_t, dtype=np.float32)
            np.hstack([cost_matrix, augmentation])

        r_aug = benchmark(augment_only, n_warmup=5, n_runs=100)
        r_full = benchmark(lambda: hungarian(cost_matrix), n_warmup=3, n_runs=50)

        pct = r_aug.mean_ms / max(r_full.mean_ms, 0.001) * 100
        rows.append([name, f'{n_t}x{n_d}',
                     f'{r_aug.mean_ms:.4f}', f'{r_full.mean_ms:.3f}', f'{pct:.1f}%'])

    print_table('Augmentation Overhead', headers, rows)


def verify_correctness() -> None:
    """Verify that scipy_direct produces equivalent results to hungarian."""
    cost_matrix = generate_cost_matrix(30, 40, sparsity=0.3, seed=99)

    matches_h, ut_h, ud_h = hungarian(cost_matrix)
    matches_d, ut_d, ud_d = scipy_direct(cost_matrix)

    cost_h = sum(cost_matrix[t, d] for t, d in matches_h)
    cost_d = sum(cost_matrix[t, d] for t, d in matches_d)

    print(f'\nCorrectness check (30x40 matrix):')
    print(f'  hungarian matches={len(matches_h)}, total cost={cost_h:.4f}')
    print(f'  scipy_direct matches={len(matches_d)}, total cost={cost_d:.4f}')
    print(f'  Cost difference: {abs(cost_h - cost_d):.6f}')


if __name__ == '__main__':
    profile_assignment()
    profile_augmentation_overhead()
    verify_correctness()
