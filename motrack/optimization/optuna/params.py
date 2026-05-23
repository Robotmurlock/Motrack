"""
Per-sampler params dataclasses for the Optuna pipeline family.

Each sampler registered with :func:`pipeline_factory` declares an explicit
allowlist of accepted ``sampler_params`` keys via a dataclass. Anything
else surfaces as ``TypeError`` at construction time instead of silently
no-opping inside Optuna's sampler kwargs.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class HyperbandPrunerConfig:
    """Optuna ``HyperbandPruner`` settings + scene-count rung schedule.

    Rung k uses ``min_resource * reduction_factor**k`` scenes, clipped to
    ``max_resource``. With defaults `(3, 25, 3)` the rungs are `[3, 9, 25]`.
    The trial reports its HOTA at each rung and the pruner decides whether
    to advance — bad trials get killed at low fidelity.

    A non-None pruner config switches the Optuna pipeline into rung-based
    evaluation mode; without it, every trial does one full-fidelity eval.
    """
    min_resource: int = 3
    max_resource: int = 25
    reduction_factor: int = 3
    seed: Optional[int] = None


@dataclass
class RandomParams:
    """``sampler='random'`` — Optuna ``RandomSampler``."""
    seed: Optional[int] = None
    pruner: Optional[Dict[str, Any]] = None  # coerced to HyperbandPrunerConfig


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
    pruner: Optional[Dict[str, Any]] = None


@dataclass
class WarmTPEParams(TPEParams):
    """``sampler='warm_tpe'`` — same kwargs as :class:`TPEParams`.

    Distinct dataclass kept for symmetry with the registry shape and so
    sampler-specific defaults can diverge later without churning callers.
    """


@dataclass
class GPParams:
    """``sampler='gp'`` — Optuna ``GPSampler`` (Gaussian-Process BO).

    Subset of GPSampler kwargs that we expose; defaults match Optuna's
    until proven otherwise. Optuna's GPSampler does its own random
    warm-up before the GP kicks in (``n_startup_trials``).
    """
    n_startup_trials: int = 10
    seed: Optional[int] = None
    deterministic_objective: bool = False


@dataclass
class WarmGPParams(GPParams):
    """``sampler='warm_gp'`` — GPSampler with the manually-tuned default
    config enqueued as the first trial.
    """
