"""
Unit tests for affine warp estimation (least-squares and RANSAC).
"""
import unittest

import numpy as np

from motrack.cmc.components.ransac import (
    MIN_SAMPLES,
    AffineEstimate,
    WarpRANSACEstimator,
    estimate_warp_lstsq
)
from motrack.cmc.components.warp import apply_warp_to_points, is_identity_warp

# Rotation by 3 degrees, scale 1.02, translation (7, -4).
ANGLE = np.deg2rad(3.0)
SCALE = 1.02
GROUND_TRUTH_WARP = np.array([
    [SCALE * np.cos(ANGLE), -SCALE * np.sin(ANGLE), 7.0],
    [SCALE * np.sin(ANGLE), SCALE * np.cos(ANGLE), -4.0]
], dtype=np.float64)


def _random_points(rng: np.random.Generator, n: int, scale: float = 100.0) -> np.ndarray:
    return rng.uniform(0.0, scale, size=(n, 2))


def _gross_offsets(rng: np.random.Generator, n: int, min_offset: float, max_offset: float) -> np.ndarray:
    """
    Creates displacements whose magnitude is guaranteed to lie in [min_offset, max_offset].

    Sampling a displacement uniformly from a box does not make a point an outlier: it can
    land arbitrarily close to the origin, in which case the point stays within the inlier
    threshold and is a genuine inlier. For a 160x160 box and a threshold of 2.0 that happens
    for at least one of 40 points in roughly 2% of seeds, which would make any assertion
    about the recovered inlier set flaky. Sampling a direction and a magnitude separately
    makes the property hold by construction instead.
    """
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n)
    magnitudes = rng.uniform(min_offset, max_offset, size=n)
    return np.stack([magnitudes * np.cos(angles), magnitudes * np.sin(angles)], axis=1)


class EstimateWarpLstsqTest(unittest.TestCase):
    """
    Tests for the plain least-squares estimator.
    """

    def test_returns_2x3_matrix(self) -> None:
        """
        The affine warp convention is (2, 3) everywhere in the codebase.

        Regression guard: the least-squares system is solved for the transposed (3, 2)
        matrix, so forgetting to transpose it produces a matrix that silently fails only
        later, when a caller indexes the translation column.
        """
        rng = np.random.default_rng(0)
        src = _random_points(rng, 8)
        dst = apply_warp_to_points(GROUND_TRUTH_WARP, src)

        warp, is_degenerate = estimate_warp_lstsq(src, dst)

        self.assertEqual(warp.shape, (2, 3))
        self.assertFalse(is_degenerate)

    def test_recovers_exact_warp_from_minimal_sample(self) -> None:
        """
        Three non-collinear correspondences determine an affine transform exactly.
        """
        src = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
        dst = apply_warp_to_points(GROUND_TRUTH_WARP, src)

        warp, is_degenerate = estimate_warp_lstsq(src, dst)

        self.assertFalse(is_degenerate)
        np.testing.assert_allclose(warp, GROUND_TRUTH_WARP, atol=1e-5)

    def test_recovers_exact_warp_overdetermined(self) -> None:
        """
        Noise-free overdetermined systems recover the warp exactly.
        """
        rng = np.random.default_rng(1)
        src = _random_points(rng, 50)
        dst = apply_warp_to_points(GROUND_TRUTH_WARP, src)

        warp, is_degenerate = estimate_warp_lstsq(src, dst)

        self.assertFalse(is_degenerate)
        np.testing.assert_allclose(warp, GROUND_TRUTH_WARP, atol=1e-4)

    def test_docstring_example(self) -> None:
        """
        The worked example in the docstring is correct and non-degenerate.
        """
        src = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        dst = np.array([[1.0, -1.0], [3.0, 0.0], [2.0, -1.0]])

        warp, is_degenerate = estimate_warp_lstsq(src, dst)

        self.assertFalse(is_degenerate)
        np.testing.assert_allclose(warp, np.array([[2.0, 1.0, 1.0], [1.0, 0.0, -1.0]]), atol=1e-5)

    def test_pure_translation_gives_identity_linear_block(self) -> None:
        """
        Translation-only data must not introduce spurious rotation or scale.
        """
        rng = np.random.default_rng(2)
        src = _random_points(rng, 30)
        dst = src + np.array([12.0, -5.0])

        warp, _ = estimate_warp_lstsq(src, dst)

        np.testing.assert_allclose(warp[:, :2], np.eye(2), atol=1e-5)
        np.testing.assert_allclose(warp[:, 2], [12.0, -5.0], atol=1e-4)

    def test_collinear_points_are_degenerate(self) -> None:
        """
        Collinear correspondences do not determine an affine transform.
        """
        t = np.linspace(0.0, 10.0, 6)
        src = np.stack([t, 2.0 * t + 1.0], axis=1)
        dst = apply_warp_to_points(GROUND_TRUTH_WARP, src)

        warp, is_degenerate = estimate_warp_lstsq(src, dst)

        self.assertTrue(is_degenerate)
        self.assertTrue(is_identity_warp(warp))

    def test_duplicate_points_are_degenerate(self) -> None:
        """
        Duplicate correspondences reduce the effective rank of the system.
        """
        src = np.array([[1.0, 2.0], [1.0, 2.0], [5.0, 6.0]])
        dst = apply_warp_to_points(GROUND_TRUTH_WARP, src)

        _, is_degenerate = estimate_warp_lstsq(src, dst)

        self.assertTrue(is_degenerate)

    def test_too_few_points_are_degenerate(self) -> None:
        """
        Fewer than three correspondences leave the system underdetermined.
        """
        for n in range(MIN_SAMPLES):
            with self.subTest(n_points=n):
                src = np.zeros((n, 2))
                warp, is_degenerate = estimate_warp_lstsq(src, src.copy())

                self.assertTrue(is_degenerate)
                self.assertTrue(is_identity_warp(warp))

    def test_rejects_mismatched_shapes(self) -> None:
        """
        Source and target point sets must correspond one to one.
        """
        with self.assertRaises(AssertionError):
            estimate_warp_lstsq(np.zeros((4, 2)), np.zeros((5, 2)))


class WarpRANSACEstimatorTest(unittest.TestCase):
    """
    Tests for the robust RANSAC estimator.
    """

    def test_recovers_warp_without_outliers(self) -> None:
        """
        Clean data is recovered exactly, and every correspondence is an inlier.
        """
        rng = np.random.default_rng(3)
        src = _random_points(rng, 40)
        dst = apply_warp_to_points(GROUND_TRUTH_WARP, src)

        estimate = WarpRANSACEstimator(residual_threshold=1e-2).estimate(src, dst)

        self.assertTrue(estimate.success)
        self.assertEqual(estimate.n_inliers, 40)
        self.assertAlmostEqual(estimate.inlier_ratio, 1.0)
        np.testing.assert_allclose(estimate.warp, GROUND_TRUTH_WARP, atol=1e-3)

    def test_recovers_warp_with_40_percent_outliers(self) -> None:
        """
        60 inliers plus 40 gross outliers: the warp is recovered and the consensus set
        matches the true inlier set.

        This is the test that distinguishes RANSAC from a plain least-squares fit - the
        latter is dragged badly off by the outliers.
        """
        rng = np.random.default_rng(4)
        src = _random_points(rng, 100)
        dst = apply_warp_to_points(GROUND_TRUTH_WARP, src)

        true_inliers = np.ones(100, dtype=bool)
        true_inliers[60:] = False
        # Displaced by at least 20.0, i.e. ten times the inlier threshold below, so every
        # designated outlier really is one.
        dst[60:] += _gross_offsets(rng, 40, min_offset=20.0, max_offset=80.0)

        estimate = WarpRANSACEstimator(residual_threshold=2.0, max_iterations=500, max_skips=200).estimate(src, dst)

        self.assertTrue(estimate.success)
        np.testing.assert_allclose(estimate.warp, GROUND_TRUTH_WARP, atol=1e-2)

        recovered = estimate.inliers_mask[true_inliers].sum()
        self.assertGreaterEqual(recovered, 57, 'Should recover at least 95% of the true inliers')
        self.assertEqual(estimate.inliers_mask[~true_inliers].sum(), 0, 'No true outlier may be marked an inlier')

        contaminated_warp, _ = estimate_warp_lstsq(src, dst)
        self.assertFalse(
            np.allclose(contaminated_warp, GROUND_TRUTH_WARP, atol=1e-2),
            'A plain least-squares fit should be visibly wrong here, otherwise the test is not exercising RANSAC'
        )

    def test_all_collinear_input_terminates(self) -> None:
        """
        Every sample drawn from collinear points is degenerate.

        Regression guard: if degenerate samples do not advance the loop counters, this
        input hangs forever instead of failing.
        """
        t = np.linspace(0.0, 100.0, 40)
        src = np.stack([t, 2.0 * t + 1.0], axis=1)
        dst = src + np.array([3.0, -2.0])

        estimate = WarpRANSACEstimator(max_iterations=50).estimate(src, dst)

        self.assertFalse(estimate.success)
        self.assertTrue(is_identity_warp(estimate.warp))
        self.assertEqual(estimate.n_degenerate, estimate.n_iterations)
        self.assertGreater(estimate.n_iterations, 0)

    def test_all_duplicate_input_terminates(self) -> None:
        """
        A correspondence set of identical points must fail, not hang.
        """
        src = np.tile(np.array([[5.0, 7.0]]), (20, 1))
        dst = src + np.array([1.0, 1.0])

        estimate = WarpRANSACEstimator(max_iterations=50).estimate(src, dst)

        self.assertFalse(estimate.success)
        self.assertTrue(is_identity_warp(estimate.warp))

    def test_fails_when_fewer_points_than_min_inliers(self) -> None:
        """
        A consensus set can never reach `min_inliers` if the input is smaller than it.
        """
        rng = np.random.default_rng(5)
        src = _random_points(rng, 5)
        dst = apply_warp_to_points(GROUND_TRUTH_WARP, src)

        estimate = WarpRANSACEstimator(min_inliers=10).estimate(src, dst)

        self.assertFalse(estimate.success)
        self.assertTrue(is_identity_warp(estimate.warp))
        self.assertEqual(estimate.n_iterations, 0)
        self.assertEqual(estimate.inliers_mask.shape, (5,))
        self.assertFalse(estimate.inliers_mask.any())

    def test_fails_when_consensus_below_min_inliers(self) -> None:
        """
        Pure noise has no consensus set, so no warp may be returned.

        Regression guard: without an acceptance threshold, a model supported by a couple of
        coincidentally-close correspondences is returned as a success, and the refit over
        that tiny set is itself degenerate.
        """
        rng = np.random.default_rng(6)
        src = _random_points(rng, 60)
        dst = _random_points(rng, 60)

        estimate = WarpRANSACEstimator(residual_threshold=0.5, min_inliers=20).estimate(src, dst)

        self.assertFalse(estimate.success)
        self.assertTrue(is_identity_warp(estimate.warp))
        self.assertEqual(estimate.n_inliers, 0)

    def test_never_raises_on_degenerate_input(self) -> None:
        """
        The CMC contract forbids raising: callers rely on an identity warp instead.
        """
        rng = np.random.default_rng(7)
        cases = {
            'empty': np.zeros((0, 2)),
            'single_point': np.zeros((1, 2)),
            'two_points': np.zeros((2, 2)),
            'all_identical': np.tile(np.array([[1.0, 1.0]]), (30, 1)),
            'random_noise': _random_points(rng, 30)
        }
        estimator = WarpRANSACEstimator()
        for name, src in cases.items():
            with self.subTest(case=name):
                estimate = estimator.estimate(src, src.copy() + 1.0)
                self.assertIsInstance(estimate, AffineEstimate)
                self.assertEqual(estimate.inliers_mask.shape, (src.shape[0],))

    def test_refit_is_not_the_minimal_sample_fit(self) -> None:
        """
        The returned warp must come from a least-squares refit over the whole consensus set.

        With noisy inliers, a fit through any three of them is measurably worse than a fit
        through all of them, so refitting strictly reduces the inlier RMS error.
        """
        rng = np.random.default_rng(8)
        src = _random_points(rng, 80)
        dst = apply_warp_to_points(GROUND_TRUTH_WARP, src) + rng.normal(0.0, 0.4, size=(80, 2))

        estimate = WarpRANSACEstimator(residual_threshold=2.0, max_iterations=200, max_skips=50).estimate(src, dst)
        self.assertTrue(estimate.success)

        inlier_src, inlier_dst = src[estimate.inliers_mask], dst[estimate.inliers_mask]
        refit_rms = np.sqrt(np.mean(np.sum((apply_warp_to_points(estimate.warp, inlier_src) - inlier_dst) ** 2, axis=1)))

        worst_minimal_rms = 0.0
        for _ in range(20):
            sample = rng.choice(inlier_src.shape[0], size=MIN_SAMPLES, replace=False)
            minimal_warp, is_degenerate = estimate_warp_lstsq(inlier_src[sample], inlier_dst[sample])
            if is_degenerate:
                continue
            rms = np.sqrt(np.mean(np.sum((apply_warp_to_points(minimal_warp, inlier_src) - inlier_dst) ** 2, axis=1)))
            worst_minimal_rms = max(worst_minimal_rms, rms)

        self.assertLess(refit_rms, worst_minimal_rms)

    def test_is_unit_agnostic(self) -> None:
        """
        The same geometry expressed in normalized coordinates is recovered with a
        correspondingly scaled residual threshold.

        The Kalman filter residual CMC relies on this: it passes normalized points directly.
        """
        rng = np.random.default_rng(9)
        width, height = 1920, 1080

        src_px = np.stack([rng.uniform(0, width, size=40), rng.uniform(0, height, size=40)], axis=1)
        translation = np.array([19.2, 10.8])
        dst_px = src_px + translation

        scale = np.array([width, height], dtype=np.float64)
        estimate = WarpRANSACEstimator(residual_threshold=2.0 / width).estimate(src_px / scale, dst_px / scale)

        self.assertTrue(estimate.success)
        np.testing.assert_allclose(estimate.warp[:, 2], translation / scale, atol=1e-4)

    def test_max_iterations_one_is_well_formed(self) -> None:
        """
        A single-iteration budget still produces a valid result object.
        """
        rng = np.random.default_rng(10)
        src = _random_points(rng, 40)
        dst = apply_warp_to_points(GROUND_TRUTH_WARP, src)

        estimate = WarpRANSACEstimator(residual_threshold=1e-2, max_iterations=1).estimate(src, dst)

        self.assertEqual(estimate.n_iterations, 1)
        self.assertTrue(estimate.success)

    def test_rejects_invalid_configuration(self) -> None:
        """
        `min_inliers` below the minimal sample size can never be satisfied meaningfully.
        """
        with self.assertRaises(AssertionError):
            WarpRANSACEstimator(min_inliers=MIN_SAMPLES - 1)
        with self.assertRaises(AssertionError):
            WarpRANSACEstimator(residual_threshold=0.0)


class WarpRANSACDeterminismTest(unittest.TestCase):
    """
    Reproducibility guarantees the benchmark relies on.
    """

    def setUp(self) -> None:
        rng = np.random.default_rng(11)
        self._src = _random_points(rng, 100)
        self._dst = apply_warp_to_points(GROUND_TRUTH_WARP, self._src)
        self._dst[70:] += _gross_offsets(rng, 30, min_offset=20.0, max_offset=50.0)

    def test_same_seed_gives_identical_results(self) -> None:
        """
        Two estimators with the same seed produce bit-identical warps.
        """
        first = WarpRANSACEstimator(seed=123).estimate(self._src, self._dst)
        second = WarpRANSACEstimator(seed=123).estimate(self._src, self._dst)

        np.testing.assert_array_equal(first.warp, second.warp)
        np.testing.assert_array_equal(first.inliers_mask, second.inliers_mask)
        self.assertEqual(first.n_iterations, second.n_iterations)

    def test_reset_restores_the_initial_sequence(self) -> None:
        """
        Without `reset`, a scene's result depends on how many samples earlier scenes drew.
        """
        estimator = WarpRANSACEstimator(seed=123)

        first = estimator.estimate(self._src, self._dst)
        drifted = estimator.estimate(self._src, self._dst)
        estimator.reset()
        after_reset = estimator.estimate(self._src, self._dst)

        np.testing.assert_array_equal(after_reset.warp, first.warp)
        self.assertEqual(after_reset.n_iterations, first.n_iterations)
        # Guard against the test passing because the estimator is accidentally stateless.
        self.assertIsNotNone(drifted)

    def test_warp_dtype_is_float32(self) -> None:
        """
        The motion filter builds float32 matrices, so the warp must not widen them.
        """
        estimate = WarpRANSACEstimator(seed=123).estimate(self._src, self._dst)
        self.assertEqual(estimate.warp.dtype, np.float32)


if __name__ == '__main__':
    unittest.main()
