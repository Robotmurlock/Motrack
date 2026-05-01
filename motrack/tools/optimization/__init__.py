"""
Tracker hyperparameter optimization.

``run_optimize`` dispatches to the algorithm selected by
``cfg.optimizer.sampler``:

- ``'random' | 'tpe' | 'warm_tpe'`` — Optuna-driven (``tpe.run_tpe``).
- ``'mfgcs'`` — Multi-Fidelity Greedy Coordinate Search.

``OptimizationResults`` and ``TrialResult`` are the typed schemas for
``optimization_results.json``. Dataset construction is pluggable via
``dataset_builder``.
"""
from motrack.config_parser import GlobalConfig
from motrack.tools.dataset_builder import DatasetBuilder, default_dataset_builder
from motrack.tools.optimization.results import OptimizationResults, TrialResult
from motrack.tools.optimization import tpe


def run_optimize(
    cfg: GlobalConfig,
    dataset_builder: DatasetBuilder = default_dataset_builder,
) -> None:
    """Top-level optimization entry point.

    Validates the optimizer config and dispatches to the algorithm selected
    by ``cfg.optimizer.sampler``.
    """
    assert cfg.optimizer is not None, 'optimizer config is required'
    sampler = cfg.optimizer.sampler
    if sampler in ('random', 'tpe', 'warm_tpe'):
        return tpe.run_tpe(cfg, dataset_builder=dataset_builder)
    if sampler == 'mfgcs':
        from motrack.tools.optimization.mfgcs import MFGCSAlgorithm
        return MFGCSAlgorithm(cfg, dataset_builder=dataset_builder).run()
    raise ValueError(f'Unknown sampler: {sampler}')


__all__ = [
    'run_optimize',
    'OptimizationResults',
    'TrialResult',
]
