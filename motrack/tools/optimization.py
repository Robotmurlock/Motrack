"""
HPO CLI driver.

This module is intentionally thin: the algorithms, factories, params
dataclasses, and result schemas live in :mod:`motrack.optimization`.
The driver here just resolves the registered pipeline for
``cfg.optimizer.sampler`` and runs it end-to-end.
"""
from motrack.config_parser import GlobalConfig
from motrack.optimization import (
    OptimizationResults,
    TrialResult,
    pipeline_factory,
)
from motrack.tools.dataset_builder import DatasetBuilder, default_dataset_builder


def run_optimize(
    cfg: GlobalConfig,
    dataset_builder: DatasetBuilder = default_dataset_builder,
) -> None:
    """Top-level optimization entry point.

    Validates the optimizer config and dispatches via :func:`pipeline_factory`.
    """
    assert cfg.optimizer is not None, 'optimizer config is required'
    pipeline = pipeline_factory(
        cfg.optimizer.sampler,
        cfg.optimizer.sampler_params,
        cfg,
        dataset_builder,
    )
    pipeline.run()


__all__ = [
    'run_optimize',
    'OptimizationResults',
    'TrialResult',
]
