"""
Coordinate optimizers for MFGCS.

Each ``CoordinateOptimizer`` searches a single search-space parameter while
all others are held fixed. Three variants are shipped:

- ``RandomCoordinateOptimizer`` — unbiased baseline.
- ``CoarseToFineCoordinateOptimizer`` — robust default for ordered params.
- ``TernaryCoordinateOptimizer`` — efficient on smooth continuous floats.

Variants gracefully fall back when a parameter type does not match their
assumptions (e.g. ternary on a categorical → random; coarse-to-fine on a
categorical → enumeration).
"""
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from motrack.config_parser import FactorySpec, SearchSpaceParam

logger = logging.getLogger('MFGCS-Coordinate')

LowEval = Callable[[Any], float]


# ---------------------------------------------------------------------------
# Search window — current effective bounds/choices for one parameter
# ---------------------------------------------------------------------------

@dataclass
class SearchWindow:
    """Mutable per-parameter window narrowed by the shrinker after accepts.

    For numeric (int/float) params, ``low``/``high`` are bounds. For
    categorical params, ``choices`` is the active subset. Dependent-bound
    constraints (``min_param``/``max_param``) are resolved by the algorithm
    at evaluation time, not stored here.
    """
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None

    @classmethod
    def from_spec(cls, spec: SearchSpaceParam) -> 'SearchWindow':
        if spec.type == 'categorical':
            return cls(choices=list(spec.choices or []))
        return cls(low=spec.low, high=spec.high)


# ---------------------------------------------------------------------------
# Helpers for sampling within a window
# ---------------------------------------------------------------------------

def _quantize_int(value: float, step: int) -> int:
    """Round ``value`` to the nearest multiple of ``step``."""
    if step <= 1:
        return int(round(value))
    return int(round(value / step) * step)


def _is_degenerate(spec: SearchSpaceParam, window: SearchWindow) -> bool:
    """Window has at most one feasible value — caller should skip optimization."""
    if spec.type == 'categorical':
        return window.choices is not None and len(window.choices) <= 1
    if window.low is None or window.high is None:
        return False
    if spec.type == 'int':
        return int(window.low) >= int(window.high)
    return float(window.low) >= float(window.high)


def _sample_uniform(spec: SearchSpaceParam, low: float, high: float, rng: np.random.Generator) -> Any:
    """Draw one value from a numeric parameter within ``[low, high]``."""
    if spec.type == 'int':
        step = int(spec.step) if spec.step else 1
        lo, hi = int(low), int(high)
        if lo >= hi:
            return lo
        # Honor step by sampling on the reduced lattice.
        if step > 1:
            n = (hi - lo) // step
            k = int(rng.integers(0, n + 1))
            return lo + k * step
        return int(rng.integers(lo, hi + 1))
    # float
    if spec.log:
        if low <= 0 or high <= 0:
            raise ValueError(f'log-scale float requires positive bounds, got [{low}, {high}]')
        u = rng.uniform(math.log(low), math.log(high))
        v = math.exp(u)
    else:
        v = float(rng.uniform(low, high))
    if spec.step:
        # Snap to step grid, anchored at low.
        v = low + round((v - low) / spec.step) * spec.step
    return float(min(max(v, low), high))


def _grid_points(spec: SearchSpaceParam, low: float, high: float, n: int) -> List[Any]:
    """Build ``n`` evenly spaced grid points across ``[low, high]``.

    Honors log-scale for floats and step quantization for ints. Always
    includes the endpoints. Duplicates (after quantization) are removed
    while preserving order.
    """
    if n < 2:
        n = 2
    if spec.type == 'int':
        lo, hi = int(low), int(high)
        if lo >= hi:
            return [lo]
        step = int(spec.step) if spec.step else 1
        raw = np.linspace(lo, hi, n)
        out: List[int] = []
        for r in raw:
            q = _quantize_int(float(r), step)
            q = max(lo, min(hi, q))
            if not out or q != out[-1]:
                out.append(q)
        return out
    # float
    if spec.log:
        if low <= 0 or high <= 0:
            raise ValueError(f'log-scale float requires positive bounds, got [{low}, {high}]')
        raw = np.exp(np.linspace(math.log(low), math.log(high), n))
    else:
        raw = np.linspace(low, high, n)
    out_f: List[float] = []
    for r in raw:
        v = float(r)
        if spec.step:
            v = low + round((v - low) / spec.step) * spec.step
        v = float(min(max(v, low), high))
        if not out_f or v != out_f[-1]:
            out_f.append(v)
    return out_f


# ---------------------------------------------------------------------------
# CoordinateOptimizer ABC + variants
# ---------------------------------------------------------------------------

class CoordinateOptimizer(ABC):
    """Search a single parameter's slice of the objective."""

    @abstractmethod
    def optimize(
        self,
        spec: SearchSpaceParam,
        current_value: Any,
        low_eval: LowEval,
        *,
        window: SearchWindow,
    ) -> Any:
        """Return the best candidate value found within ``window``.

        ``low_eval`` is a closure that takes a candidate value and returns
        its low-fidelity score (higher is better). The returned value MUST
        be feasible within ``window``; if no improvement was found the
        optimizer should return ``current_value`` so the caller can skip
        the high-fidelity check.
        """


@dataclass
class _RandomCoordinateOptimizerParams:
    n_candidates: int = 5
    seed: int = 42


class RandomCoordinateOptimizer(CoordinateOptimizer):
    """Evaluate ``n_candidates`` uniform-random samples; return the best.

    For categorical params, samples without replacement up to the size of
    the choice set.
    """

    def __init__(self, n_candidates: int = 5, seed: int = 42) -> None:
        if n_candidates < 1:
            raise ValueError(f'n_candidates must be >= 1, got {n_candidates}')
        self._n = n_candidates
        self._rng = np.random.default_rng(seed)

    def optimize(self, spec, current_value, low_eval, *, window) -> Any:
        if _is_degenerate(spec, window):
            return current_value

        candidates: List[Any]
        if spec.type == 'categorical':
            choices = list(window.choices or [])
            n = min(self._n, len(choices))
            idx = self._rng.choice(len(choices), size=n, replace=False)
            candidates = [choices[int(i)] for i in idx]
        else:
            candidates = [
                _sample_uniform(spec, float(window.low), float(window.high), self._rng)
                for _ in range(self._n)
            ]

        return _pick_best(current_value, candidates, low_eval)


@dataclass
class _CoarseToFineCoordinateOptimizerParams:
    grid: int = 5
    rounds: int = 3


class CoarseToFineCoordinateOptimizer(CoordinateOptimizer):
    """Repeated grid search with the interval shrunk around each round's best.

    For categorical params with no natural ordering, falls back to a single
    full enumeration (no shrinking).
    """

    def __init__(self, grid: int = 5, rounds: int = 3) -> None:
        if grid < 2:
            raise ValueError(f'grid must be >= 2, got {grid}')
        if rounds < 1:
            raise ValueError(f'rounds must be >= 1, got {rounds}')
        self._grid = grid
        self._rounds = rounds

    def optimize(self, spec, current_value, low_eval, *, window) -> Any:
        if _is_degenerate(spec, window):
            return current_value

        if spec.type == 'categorical':
            return _pick_best(current_value, list(window.choices or []), low_eval)

        low = float(window.low)
        high = float(window.high)
        best_value: Any = current_value
        best_score: Optional[float] = None

        for round_idx in range(self._rounds):
            points = _grid_points(spec, low, high, self._grid)
            cand_value, cand_score = _eval_points(points, low_eval)
            if best_score is None or (cand_score is not None and cand_score > best_score):
                best_value, best_score = cand_value, cand_score
            # Shrink interval to ±1 grid step around best_value (in current scale).
            if len(points) <= 1:
                break
            step = (high - low) / (self._grid - 1)
            new_low = max(window.low, float(best_value) - step)
            new_high = min(window.high, float(best_value) + step)
            if not (new_high > new_low):
                break
            low, high = new_low, new_high

        if spec.type == 'int':
            return int(best_value)
        return best_value


@dataclass
class _TernaryCoordinateOptimizerParams:
    n_steps: int = 5


class TernaryCoordinateOptimizer(CoordinateOptimizer):
    """Ternary search for continuous floats.

    Falls back to ``CoarseToFineCoordinateOptimizer`` for ints (small lattice
    where ternary section is unreliable) and to ``RandomCoordinateOptimizer``
    for categoricals (no order). The fallbacks share this optimizer's
    ``n_steps`` budget by mapping to ``rounds`` / ``n_candidates``.
    """

    def __init__(self, n_steps: int = 5) -> None:
        if n_steps < 1:
            raise ValueError(f'n_steps must be >= 1, got {n_steps}')
        self._n_steps = n_steps

    def optimize(self, spec, current_value, low_eval, *, window) -> Any:
        if _is_degenerate(spec, window):
            return current_value

        if spec.type == 'categorical':
            logger.warning('TernaryCoordinateOptimizer: falling back to Random for categorical')
            fallback = RandomCoordinateOptimizer(n_candidates=self._n_steps)
            return fallback.optimize(spec, current_value, low_eval, window=window)

        if spec.type == 'int':
            logger.warning('TernaryCoordinateOptimizer: falling back to CoarseToFine for int')
            fallback = CoarseToFineCoordinateOptimizer(grid=4, rounds=self._n_steps)
            return fallback.optimize(spec, current_value, low_eval, window=window)

        # Continuous float — proper ternary search (in log space if requested).
        return self._ternary_float(spec, current_value, low_eval, window)

    def _ternary_float(self, spec, current_value, low_eval, window) -> float:
        low = float(window.low)
        high = float(window.high)
        log_scale = bool(spec.log)
        if log_scale:
            if low <= 0 or high <= 0:
                raise ValueError(f'log-scale float requires positive bounds, got [{low}, {high}]')
            lo, hi = math.log(low), math.log(high)
            to_param = math.exp
        else:
            lo, hi = low, high

            def to_param(x: float) -> float:
                return x

        best_value: float = float(current_value)
        best_score: Optional[float] = None
        evaluated: Dict[float, float] = {}

        def eval_at(x: float) -> float:
            v = to_param(x)
            if spec.step:
                v = low + round((v - low) / spec.step) * spec.step
            v = float(min(max(v, low), high))
            cached = evaluated.get(v)
            if cached is not None:
                return cached
            score = float(low_eval(v))
            evaluated[v] = score
            return score

        for _ in range(self._n_steps):
            if hi - lo <= 1e-12:
                break
            m1 = lo + (hi - lo) / 3.0
            m2 = hi - (hi - lo) / 3.0
            s1, s2 = eval_at(m1), eval_at(m2)
            if s1 < s2:
                lo = m1
                if best_score is None or s2 > best_score:
                    best_value, best_score = to_param(m2), s2
            else:
                hi = m2
                if best_score is None or s1 > best_score:
                    best_value, best_score = to_param(m1), s1

        # Always also score the midpoint of the final interval and return the best seen.
        mid_x = 0.5 * (lo + hi)
        mid_score = eval_at(mid_x)
        if best_score is None or mid_score > best_score:
            best_value = to_param(mid_x)

        if spec.step:
            best_value = low + round((best_value - low) / spec.step) * spec.step
        return float(min(max(best_value, low), high))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _eval_points(points: List[Any], low_eval: LowEval) -> Tuple[Any, Optional[float]]:
    """Evaluate every point and return ``(best_value, best_score)``."""
    best_value: Any = points[0] if points else None
    best_score: Optional[float] = None
    for p in points:
        s = float(low_eval(p))
        if best_score is None or s > best_score:
            best_value, best_score = p, s
    return best_value, best_score


def _pick_best(current_value: Any, candidates: List[Any], low_eval: LowEval) -> Any:
    """Pick the highest-scoring candidate; tie-break to ``current_value`` if present."""
    if not candidates:
        return current_value
    best_value, best_score = _eval_points(candidates, low_eval)
    if current_value in candidates:
        return best_value
    # Compare against current_value too so we never make a strict downgrade at this stage.
    cur_score = float(low_eval(current_value))
    if best_score is None or cur_score >= best_score:
        return current_value
    return best_value


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_COORDINATE_REGISTRY: Dict[str, Any] = {
    'random': (RandomCoordinateOptimizer, _RandomCoordinateOptimizerParams),
    'coarse_to_fine': (CoarseToFineCoordinateOptimizer, _CoarseToFineCoordinateOptimizerParams),
    'ternary': (TernaryCoordinateOptimizer, _TernaryCoordinateOptimizerParams),
}


def coordinate_optimizer_factory(spec: FactorySpec) -> CoordinateOptimizer:
    """Construct a coordinate optimizer from a ``FactorySpec`` declaration.

    Raises:
        ValueError: if ``spec.type`` is not registered.
        TypeError: if ``spec.params`` contains keys the variant doesn't accept.
    """
    if spec.type not in _COORDINATE_REGISTRY:
        raise ValueError(
            f'Unknown coordinate optimizer type: "{spec.type}". '
            f'Known: {sorted(_COORDINATE_REGISTRY)}'
        )
    cls, params_cls = _COORDINATE_REGISTRY[spec.type]
    typed = params_cls(**(spec.params or {}))
    return cls(**typed.__dict__)
