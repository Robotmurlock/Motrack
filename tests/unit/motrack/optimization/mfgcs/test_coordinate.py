"""
Tests for ``motrack.optimization.mfgcs.coordinate``.
"""
import unittest

from motrack.config_parser import FactorySpec, SearchSpaceParam
from motrack.optimization.mfgcs.coordinate import (
    GridCoordinateOptimizer,
    CoordinateOptimizer,
    RandomCoordinateOptimizer,
    SearchWindow,
    TernaryCoordinateOptimizer,
    coordinate_optimizer_factory,
)


def _quadratic(peak: float):
    """Unimodal objective with maximum at ``peak``."""
    return lambda x: -((float(x) - peak) ** 2)


class CoordinateOptimizerFloatTest(unittest.TestCase):
    """All variants find the maximum of a unimodal float objective."""

    def setUp(self) -> None:
        self.spec = SearchSpaceParam(type='float', low=0.0, high=1.0)
        self.window = SearchWindow.from_spec(self.spec)
        self.peak = 0.7
        self.f = _quadratic(self.peak)

    def test_random_finds_peak_within_tolerance(self) -> None:
        opt = RandomCoordinateOptimizer(n_candidates=40, seed=0)
        best = opt.optimize(self.spec, current_value=0.1, low_eval=self.f, window=self.window)
        self.assertLess(abs(best - self.peak), 0.1)

    def test_grid_converges(self) -> None:
        opt = GridCoordinateOptimizer(grid=5, rounds=4)
        best = opt.optimize(self.spec, current_value=0.1, low_eval=self.f, window=self.window)
        self.assertLess(abs(best - self.peak), 0.05)

    def test_ternary_converges(self) -> None:
        opt = TernaryCoordinateOptimizer(n_steps=10)
        best = opt.optimize(self.spec, current_value=0.1, low_eval=self.f, window=self.window)
        self.assertLess(abs(best - self.peak), 0.02)


class CoordinateOptimizerIntTest(unittest.TestCase):
    """Integer params: grid and ternary fallback both work."""

    def setUp(self) -> None:
        self.spec = SearchSpaceParam(type='int', low=1, high=20)
        self.window = SearchWindow.from_spec(self.spec)
        self.target = 13
        self.f = _quadratic(self.target)

    def test_grid_int(self) -> None:
        opt = GridCoordinateOptimizer(grid=5, rounds=4)
        best = opt.optimize(self.spec, current_value=1, low_eval=self.f, window=self.window)
        self.assertEqual(best, self.target)
        self.assertIsInstance(best, int)

    def test_ternary_falls_back_for_int(self) -> None:
        opt = TernaryCoordinateOptimizer(n_steps=4)
        best = opt.optimize(self.spec, current_value=1, low_eval=self.f, window=self.window)
        self.assertLessEqual(abs(best - self.target), 2)


class CoordinateOptimizerCategoricalTest(unittest.TestCase):
    """Categorical params: ternary/grid fall back; random enumerates."""

    def setUp(self) -> None:
        self.spec = SearchSpaceParam(type='categorical', choices=['a', 'b', 'c', 'd', 'e'])
        self.window = SearchWindow.from_spec(self.spec)
        self.scores = {'a': 0.1, 'b': 0.2, 'c': 0.9, 'd': 0.4, 'e': 0.3}

    def test_random_picks_best(self) -> None:
        opt = RandomCoordinateOptimizer(n_candidates=5, seed=0)
        best = opt.optimize(self.spec, current_value='a', low_eval=lambda v: self.scores[v], window=self.window)
        self.assertEqual(best, 'c')

    def test_grid_enumerates_all(self) -> None:
        opt = GridCoordinateOptimizer(grid=5, rounds=2)
        best = opt.optimize(self.spec, current_value='a', low_eval=lambda v: self.scores[v], window=self.window)
        self.assertEqual(best, 'c')

    def test_ternary_falls_back_to_random(self) -> None:
        opt = TernaryCoordinateOptimizer(n_steps=10)
        best = opt.optimize(self.spec, current_value='a', low_eval=lambda v: self.scores[v], window=self.window)
        # With 10 candidates against 5 unique choices, the best one is reliably found.
        self.assertEqual(best, 'c')


class CoordinateOptimizerEdgeCasesTest(unittest.TestCase):
    """Degenerate windows and log-scale handling."""

    def test_degenerate_float_returns_current(self) -> None:
        spec = SearchSpaceParam(type='float', low=0.5, high=0.5)
        window = SearchWindow(low=0.5, high=0.5)
        opt = GridCoordinateOptimizer(grid=5, rounds=3)
        called = []
        best = opt.optimize(spec, current_value=0.5, low_eval=lambda v: called.append(v) or 0.0, window=window)
        self.assertEqual(best, 0.5)
        self.assertEqual(len(called), 0, 'no eval should run for a degenerate window')

    def test_single_choice_categorical_returns_current(self) -> None:
        spec = SearchSpaceParam(type='categorical', choices=['only'])
        window = SearchWindow.from_spec(spec)
        opt = RandomCoordinateOptimizer(n_candidates=3, seed=0)
        best = opt.optimize(spec, current_value='only', low_eval=lambda v: 1.0, window=window)
        self.assertEqual(best, 'only')

    def test_log_scale_float_finds_peak(self) -> None:
        spec = SearchSpaceParam(type='float', low=1.0, high=1000.0, log=True)
        window = SearchWindow.from_spec(spec)
        peak = 50.0
        opt = TernaryCoordinateOptimizer(n_steps=20)
        best = opt.optimize(spec, current_value=1.0, low_eval=_quadratic(peak), window=window)
        self.assertLess(abs(best - peak), peak * 0.1)


class CoordinateOptimizerFactoryTest(unittest.TestCase):
    """Factory dispatch + validation."""

    def test_builds_each_variant(self) -> None:
        for variant, params in (
            ('random', {'n_candidates': 3, 'seed': 1}),
            ('grid', {'grid': 4, 'rounds': 2}),
            ('ternary', {'n_steps': 5}),
        ):
            opt = coordinate_optimizer_factory(FactorySpec(type=variant, params=params))
            self.assertIsInstance(opt, CoordinateOptimizer)

    def test_unknown_type_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, 'Unknown coordinate optimizer type'):
            coordinate_optimizer_factory(FactorySpec(type='bogus', params={}))

    def test_unknown_param_key_raises(self) -> None:
        with self.assertRaises(TypeError):
            coordinate_optimizer_factory(FactorySpec(type='random', params={'bad_key': 1}))


if __name__ == '__main__':
    unittest.main()
