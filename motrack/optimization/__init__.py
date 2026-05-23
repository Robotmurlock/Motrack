"""
Tracker hyperparameter-optimization library.

Pipelines, factories, and result schemas live here. The CLI entry point
(``run_optimize``) lives in :mod:`motrack.tools.optimization`, where it
imports from this package.

Registered pipelines (resolved at import time):

- ``'random' | 'tpe' | 'warm_tpe' | 'gp' | 'warm_gp'`` — :class:`OptunaPipeline`.
- ``'mfgcs'`` — :class:`MFGCSPipeline`.
"""
from motrack.optimization.base import OptimizationPipeline
from motrack.optimization.factory import pipeline_factory, register_pipeline
from motrack.optimization.mfgcs import MFGCSParams, MFGCSPipeline
from motrack.optimization.optuna import (
    GPParams,
    OptunaPipeline,
    RandomParams,
    TPEParams,
    WarmGPParams,
    WarmTPEParams,
)
from motrack.optimization.results import OptimizationResults, TrialResult


register_pipeline('random', OptunaPipeline, RandomParams)
register_pipeline('tpe', OptunaPipeline, TPEParams)
register_pipeline('warm_tpe', OptunaPipeline, WarmTPEParams)
register_pipeline('gp', OptunaPipeline, GPParams)
register_pipeline('warm_gp', OptunaPipeline, WarmGPParams)
register_pipeline('mfgcs', MFGCSPipeline, MFGCSParams)


__all__ = [
    'pipeline_factory',
    'register_pipeline',
    'OptimizationPipeline',
    'OptunaPipeline',
    'MFGCSPipeline',
    'GPParams',
    'RandomParams',
    'TPEParams',
    'WarmGPParams',
    'WarmTPEParams',
    'MFGCSParams',
    'OptimizationResults',
    'TrialResult',
]
