"""
Tests for batch_predict correctness.

Verifies that batch_predict produces identical results to loop predict
for all filter implementations.
"""
import unittest

import numpy as np

from motrack.filter.algorithms.kalman_filter import BotSortKalmanWrapFilter, BotSortKalmanFilterConfig
from motrack.filter.algorithms.no_motion import NoMotionFilter, NoMotionFilterConfig
from motrack.library.kalman_filter.botsort_kf import BotSortKalmanFilter


def _make_kalman_states(n: int, seed: int = 42):
    rng = np.random.RandomState(seed)
    kf = BotSortKalmanFilter()
    states = []
    for _ in range(n):
        measurement = np.array([
            rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8),
            rng.uniform(0.02, 0.15), rng.uniform(0.02, 0.15),
        ], dtype=np.float64)
        mean, cov = kf.initiate(measurement)
        for _ in range(rng.randint(1, 5)):
            mean, cov = kf.predict(mean, cov)
        states.append((mean, cov))
    return states


class TestBotSortBatchPredict(unittest.TestCase):

    def _assert_batch_matches_loop(self, n: int, seed: int = 42):
        smf = BotSortKalmanWrapFilter(BotSortKalmanFilterConfig())
        states = _make_kalman_states(n, seed=seed)

        loop_results = [smf.predict(s) for s in states]
        loop_means = np.stack([r[0] for r in loop_results])
        loop_covs = np.stack([r[1] for r in loop_results])

        batch_results = smf.batch_predict(states)
        batch_means = np.stack([r[0] for r in batch_results])
        batch_covs = np.stack([r[1] for r in batch_results])

        np.testing.assert_allclose(batch_means, loop_means, atol=1e-10)
        np.testing.assert_allclose(batch_covs, loop_covs, atol=1e-10)

    def test_batch_10(self):
        self._assert_batch_matches_loop(10)

    def test_batch_20(self):
        self._assert_batch_matches_loop(20)

    def test_batch_50(self):
        self._assert_batch_matches_loop(50)

    def test_batch_single(self):
        self._assert_batch_matches_loop(1)

    def test_batch_empty(self):
        smf = BotSortKalmanWrapFilter(BotSortKalmanFilterConfig())
        self.assertEqual(smf.batch_predict([]), [])


class TestNoMotionBatchPredict(unittest.TestCase):

    def test_batch_returns_same_states(self):
        f = NoMotionFilter(NoMotionFilterConfig())
        rng = np.random.RandomState(42)
        states = [rng.uniform(0, 1, size=4).astype(np.float32) for _ in range(10)]

        loop_results = [f.predict(s) for s in states]
        batch_results = f.batch_predict(states)

        for loop_r, batch_r in zip(loop_results, batch_results):
            np.testing.assert_array_equal(batch_r, loop_r)

    def test_batch_empty(self):
        f = NoMotionFilter(NoMotionFilterConfig())
        self.assertEqual(f.batch_predict([]), [])


if __name__ == '__main__':
    unittest.main()
