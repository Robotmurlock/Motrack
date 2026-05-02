"""
Tests for ``motrack.optimization.mfgcs.shrinking``.
"""
import math
import unittest

from motrack.config_parser import SearchSpaceParam
from motrack.optimization.mfgcs.coordinate import SearchWindow
from motrack.optimization.mfgcs.params import MFGCSShrinkConfig
from motrack.optimization.mfgcs.shrinking import SearchSpaceShrinker


class ShrinkerTest(unittest.TestCase):
    """SearchSpaceShrinker shrinks per-type and clips to absolute bounds."""

    def test_disabled_returns_window_unchanged(self) -> None:
        spec = SearchSpaceParam(type='float', low=0.0, high=1.0)
        win = SearchWindow(low=0.0, high=1.0)
        shrinker = SearchSpaceShrinker(MFGCSShrinkConfig(enabled=False))
        out = shrinker.shrink(spec, accepted_value=0.5, window=win)
        self.assertEqual((out.low, out.high), (0.0, 1.0))

    def test_float_shrinks_around_value(self) -> None:
        spec = SearchSpaceParam(type='float', low=0.0, high=1.0)
        win = SearchWindow(low=0.0, high=1.0)
        shrinker = SearchSpaceShrinker(MFGCSShrinkConfig(radius_frac=0.25))
        out = shrinker.shrink(spec, accepted_value=0.5, window=win)
        self.assertAlmostEqual(out.low, 0.25)
        self.assertAlmostEqual(out.high, 0.75)

    def test_float_shrink_clips_to_absolute_bounds(self) -> None:
        spec = SearchSpaceParam(type='float', low=0.0, high=1.0)
        win = SearchWindow(low=0.0, high=1.0)
        shrinker = SearchSpaceShrinker(MFGCSShrinkConfig(radius_frac=0.5))
        out = shrinker.shrink(spec, accepted_value=0.05, window=win)
        self.assertAlmostEqual(out.low, 0.0)
        self.assertGreater(out.high, 0.0)

    def test_log_float_shrinks_in_log_space(self) -> None:
        spec = SearchSpaceParam(type='float', low=1.0, high=1000.0, log=True)
        win = SearchWindow(low=1.0, high=1000.0)
        shrinker = SearchSpaceShrinker(MFGCSShrinkConfig(radius_frac=0.25))
        out = shrinker.shrink(spec, accepted_value=10.0, window=win)
        # Window should still be log-symmetric-ish around 10.
        self.assertGreater(out.low, 1.0)
        self.assertLess(out.high, 1000.0)
        log_lo = math.log(out.low)
        log_hi = math.log(out.high)
        log_v = math.log(10.0)
        self.assertAlmostEqual(log_v - log_lo, log_hi - log_v, delta=0.5)

    def test_int_keeps_window_size_indices(self) -> None:
        spec = SearchSpaceParam(type='int', low=1, high=60)
        win = SearchWindow(low=1, high=60)
        shrinker = SearchSpaceShrinker(MFGCSShrinkConfig(window_size=5))
        out = shrinker.shrink(spec, accepted_value=30, window=win)
        self.assertEqual(out.low, 25)
        self.assertEqual(out.high, 35)

    def test_int_shrink_clips_to_absolute_bounds(self) -> None:
        spec = SearchSpaceParam(type='int', low=1, high=60)
        win = SearchWindow(low=1, high=60)
        shrinker = SearchSpaceShrinker(MFGCSShrinkConfig(window_size=5))
        out = shrinker.shrink(spec, accepted_value=2, window=win)
        self.assertEqual(out.low, 1)
        self.assertEqual(out.high, 7)

    def test_categorical_returns_window_unchanged(self) -> None:
        spec = SearchSpaceParam(type='categorical', choices=['a', 'b', 'c'])
        win = SearchWindow(choices=['a', 'b', 'c'])
        shrinker = SearchSpaceShrinker(MFGCSShrinkConfig())
        out = shrinker.shrink(spec, accepted_value='b', window=win)
        self.assertEqual(out.choices, ['a', 'b', 'c'])


if __name__ == '__main__':
    unittest.main()
