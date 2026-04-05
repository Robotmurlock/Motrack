"""
Profile Kalman filter operations.

Benchmarks:
1. Single operations: initiate, predict, project, update
2. Loop predict vs batch multi_predict
"""
import numpy as np

from tools.profile._common import (
    benchmark, print_table, result_cells, generate_kalman_states,
)
from motrack.library.kalman_filter.botsort_kf import BotSortKalmanFilter


def profile_single_operations() -> None:
    kf = BotSortKalmanFilter()
    rng = np.random.RandomState(42)
    measurement = np.array([0.5, 0.5, 0.1, 0.1], dtype=np.float64)

    # Initiate
    r_init = benchmark(lambda: kf.initiate(measurement), n_warmup=5, n_runs=200)

    # Predict (need a state first)
    mean, cov = kf.initiate(measurement)
    r_predict = benchmark(lambda: kf.predict(mean, cov), n_warmup=5, n_runs=200)

    # Project
    mean_hat, cov_hat = kf.predict(mean, cov)
    r_project = benchmark(lambda: kf.project(mean_hat, cov_hat), n_warmup=5, n_runs=200)

    # Update
    new_measurement = np.array([0.51, 0.51, 0.1, 0.1], dtype=np.float64)
    r_update = benchmark(lambda: kf.update(mean_hat, cov_hat, new_measurement), n_warmup=5, n_runs=200)

    headers = ['Operation', 'Mean (ms)', 'Std (ms)', 'Min (ms)', 'Median (ms)']
    rows = [
        ['initiate()'] + result_cells(r_init),
        ['predict()'] + result_cells(r_predict),
        ['project()'] + result_cells(r_project),
        ['update()'] + result_cells(r_update),
    ]
    print_table('Kalman Filter Single Operations', headers, rows)


def profile_loop_vs_batch() -> None:
    kf = BotSortKalmanFilter()

    scenarios = [10, 20, 50, 100]
    headers = ['N tracklets', 'Loop predict (ms)', 'multi_predict (ms)', 'Speedup']
    rows = []

    for n in scenarios:
        states = generate_kalman_states(n, seed=42)
        means = np.stack([s[0] for s in states])
        covs = np.stack([s[1] for s in states])

        def run_loop():
            for mean, cov in states:
                kf.predict(mean, cov)

        def run_batch():
            kf.multi_predict(means.copy(), covs.copy())

        r_loop = benchmark(run_loop, n_warmup=3, n_runs=100)
        r_batch = benchmark(run_batch, n_warmup=3, n_runs=100)

        speedup = r_loop.mean_ms / max(r_batch.mean_ms, 0.001)
        rows.append([
            str(n),
            f'{r_loop.mean_ms:.3f}',
            f'{r_batch.mean_ms:.3f}',
            f'{speedup:.2f}x',
        ])

    print_table('Loop predict() vs Batch multi_predict()', headers, rows)


def verify_correctness() -> None:
    """Verify loop predict and multi_predict produce the same results."""
    kf = BotSortKalmanFilter()
    n = 20
    states = generate_kalman_states(n, seed=42)
    means = np.stack([s[0] for s in states])
    covs = np.stack([s[1] for s in states])

    # Loop predict
    loop_means, loop_covs = [], []
    for mean, cov in states:
        m, c = kf.predict(mean, cov)
        loop_means.append(m)
        loop_covs.append(c)
    loop_means = np.stack(loop_means)
    loop_covs = np.stack(loop_covs)

    # Batch predict
    batch_means, batch_covs = kf.multi_predict(means.copy(), covs.copy())

    mean_diff = np.max(np.abs(loop_means - batch_means))
    cov_diff = np.max(np.abs(loop_covs - batch_covs))
    print(f'\nCorrectness check (N={n}):')
    print(f'  max |loop_mean - batch_mean| = {mean_diff:.2e}')
    print(f'  max |loop_cov - batch_cov|   = {cov_diff:.2e}')


if __name__ == '__main__':
    profile_single_operations()
    profile_loop_vs_batch()
    verify_correctness()
