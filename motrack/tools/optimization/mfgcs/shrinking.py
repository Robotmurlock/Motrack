"""
Search-space shrinking after accepted coordinate moves.

When a move is accepted, the parameter's window can be narrowed around the
accepted value to focus subsequent sweeps. Shape of the shrink depends on
the parameter type (numeric vs ordered-discrete vs unordered categorical)
and is configured by ``MFGCSShrinkConfig``.
"""
import logging
import math
from typing import Any

from motrack.config_parser import MFGCSShrinkConfig, SearchSpaceParam
from motrack.tools.optimization.mfgcs.coordinate import SearchWindow

logger = logging.getLogger('MFGCS-Shrink')


class SearchSpaceShrinker:
    """Narrow a parameter's window around its accepted value."""

    def __init__(self, config: MFGCSShrinkConfig) -> None:
        self._config = config

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def shrink(
        self,
        spec: SearchSpaceParam,
        accepted_value: Any,
        window: SearchWindow,
    ) -> SearchWindow:
        """Return a new window narrowed around ``accepted_value``.

        - Numeric (float): ``[v − r·(B−A), v + r·(B−A)]`` clipped to original
          static bounds; in log space when ``spec.log`` is set.
        - Numeric (int): keep ±``window_size`` indices around the accepted value.
        - Categorical: returned unchanged (no natural distance).

        ``spec.low`` / ``spec.high`` are used as the absolute clip bounds so
        that shrinking never expands beyond the original declaration.
        """
        if not self._config.enabled:
            return window

        if spec.type == 'categorical':
            return window

        if spec.type == 'int':
            return self._shrink_int(spec, int(accepted_value), window)

        return self._shrink_float(spec, float(accepted_value), window)

    def _shrink_float(self, spec: SearchSpaceParam, v: float, window: SearchWindow) -> SearchWindow:
        abs_low = float(spec.low)
        abs_high = float(spec.high)
        cur_low = float(window.low if window.low is not None else abs_low)
        cur_high = float(window.high if window.high is not None else abs_high)
        r = self._config.radius_frac

        if spec.log:
            if abs_low <= 0 or abs_high <= 0:
                logger.warning('Cannot shrink log-scale param with non-positive bounds; skipping')
                return window
            lo_log = math.log(cur_low)
            hi_log = math.log(cur_high)
            v_log = math.log(max(v, abs_low))
            span = hi_log - lo_log
            new_lo = max(math.log(abs_low), v_log - r * span)
            new_hi = min(math.log(abs_high), v_log + r * span)
            return SearchWindow(low=math.exp(new_lo), high=math.exp(new_hi))

        span = cur_high - cur_low
        new_lo = max(abs_low, v - r * span)
        new_hi = min(abs_high, v + r * span)
        if new_high_bad := new_hi <= new_lo:  # pragma: no cover — defensive
            del new_high_bad
            return window
        return SearchWindow(low=new_lo, high=new_hi)

    def _shrink_int(self, spec: SearchSpaceParam, v: int, window: SearchWindow) -> SearchWindow:
        abs_low = int(spec.low)
        abs_high = int(spec.high)
        k = self._config.window_size
        new_lo = max(abs_low, v - k)
        new_hi = min(abs_high, v + k)
        return SearchWindow(low=new_lo, high=new_hi)
