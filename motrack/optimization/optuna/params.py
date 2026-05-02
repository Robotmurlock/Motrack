"""
Per-sampler params dataclasses for the Optuna pipeline family.

Each sampler registered with :func:`pipeline_factory` declares an explicit
allowlist of accepted ``sampler_params`` keys via a dataclass. Anything
else surfaces as ``TypeError`` at construction time instead of silently
no-opping inside Optuna's sampler kwargs.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class RandomParams:
    """``sampler='random'`` — Optuna ``RandomSampler``."""
    seed: Optional[int] = None


@dataclass
class TPEParams:
    """``sampler='tpe'`` — Optuna ``TPESampler``.

    ``gamma`` is the float quantile passed to Optuna; it is rewritten to a
    callable inside :class:`OptunaPipeline` because Optuna expects a
    callable returning the lower-quantile size at each step.
    """
    gamma: Optional[float] = None
    multivariate: bool = False
    n_startup_trials: int = 10
    n_ei_candidates: int = 24
    seed: Optional[int] = None


@dataclass
class WarmTPEParams(TPEParams):
    """``sampler='warm_tpe'`` — same kwargs as :class:`TPEParams`.

    Distinct dataclass kept for symmetry with the registry shape and so
    sampler-specific defaults can diverge later without churning callers.
    """
