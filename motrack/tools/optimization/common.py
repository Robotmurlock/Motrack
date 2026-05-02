"""
Shared optimization primitives used by all algorithms.

The single primitive every algorithm needs is:

    evaluate(cfg, overrides, scenes=None) -> float

It applies parameter overrides, optionally restricts evaluation to a scene
subset (low fidelity), runs inference + eval (or loads cached results), and
returns mean HOTA. Subset and full-split evaluations get distinct on-disk
cache keys via ``cfg.hash`` because ``scene_pattern`` is part of the hash.
"""
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from motrack.common import conventions
from motrack.config_parser import GlobalConfig, SearchSpaceParam
from motrack.eval.results import EvalResults
from motrack.tools.dataset_builder import DatasetBuilder, default_dataset_builder
from motrack.tools.eval import run_eval
from motrack.tools.inference import InferenceOutputData, OptunaOutputData, run_inference

logger = logging.getLogger('Tool-Optimize')


def scenes_to_pattern(scenes: List[str]) -> str:
    """Build a strict regex matching exactly the given scene names."""
    if not scenes:
        raise ValueError('scenes must be non-empty')
    return '^(' + '|'.join(re.escape(s) for s in scenes) + ')$'


def extract_base_params(
    cfg: GlobalConfig,
    search_space: Dict[str, SearchSpaceParam],
) -> Dict[str, Any]:
    """Extract current config values for every search-space param."""
    return {dotpath: cfg.resolve(dotpath) for dotpath in search_space}


def evaluate(
    cfg: GlobalConfig,
    overrides: Dict[str, Any],
    scenes: Optional[List[str]] = None,
    *,
    optuna_data: Optional[OptunaOutputData] = None,
    dataset_builder: DatasetBuilder = default_dataset_builder,
) -> float:
    """Apply ``overrides`` and return the mean HOTA score.

    When ``scenes`` is provided, runs low-fidelity evaluation: a regex matching
    those scenes is injected as ``dataset_filter.scene_pattern`` so inference
    only tracks the subset, and ``run_eval`` is restricted to the same list.

    The on-disk eval cache (``eval_results.json`` at the trial's experiment
    path) is reused when present; ``cfg.hash`` already separates subset and
    full-split caches because it includes ``scene_pattern``.
    """
    full_overrides: Dict[str, Any] = dict(overrides)
    if scenes is not None:
        full_overrides['dataset_filter.scene_pattern'] = scenes_to_pattern(scenes)

    trial_cfg = cfg.override(full_overrides)
    trial_cfg.inference.override = True

    eval_path = conventions.get_eval_results_path(trial_cfg.experiment_path)
    if os.path.exists(eval_path):
        cached = EvalResults.load(eval_path)
        return float(np.mean(cached.combined['HOTA']['HOTA']))

    inference_output = InferenceOutputData(
        created_at=datetime.now().isoformat(),
        optuna=optuna_data,
    )
    run_inference(trial_cfg, inference_output=inference_output, dataset_builder=dataset_builder)
    results = run_eval(trial_cfg, dataset_builder=dataset_builder, scenes=scenes)
    return float(np.mean(results['combined']['HOTA']['HOTA']))


def is_eval_cached(
    cfg: GlobalConfig,
    overrides: Dict[str, Any],
    scenes: Optional[List[str]] = None,
) -> bool:
    """Probe whether :func:`evaluate` will short-circuit to the on-disk cache.

    Mirrors the path-resolution :func:`evaluate` performs. Wrapped in
    ``try / except`` so stub configs in unit tests fall through to ``False``
    rather than break the budget counters.
    """
    full_overrides: Dict[str, Any] = dict(overrides)
    if scenes is not None:
        full_overrides['dataset_filter.scene_pattern'] = scenes_to_pattern(scenes)
    try:
        trial_cfg = cfg.override(full_overrides)
        eval_path = conventions.get_eval_results_path(trial_cfg.experiment_path)
        return os.path.exists(eval_path)
    except Exception:  # pragma: no cover — defensive; stub cfgs in tests
        return False


def evaluate_with_metrics(
    cfg: GlobalConfig,
    overrides: Dict[str, Any],
    scenes: Optional[List[str]] = None,
    *,
    optuna_data: Optional[OptunaOutputData] = None,
    dataset_builder: DatasetBuilder = default_dataset_builder,
    n_full_scenes: Optional[int] = None,
) -> Tuple[float, int, float]:
    """Like :func:`evaluate` but also returns budget metrics.

    Returns ``(hota, scenes_evaluated, wall_time_s)``. ``scenes_evaluated`` is
    **zero on cache hits** so the budget axis reports only newly computed work
    — re-runs and resumed studies don't double-count previously paid scenes.
    ``wall_time_s`` is the actual elapsed time around the call (cache hits
    naturally measure as near-zero).
    """
    cache_hit = is_eval_cached(cfg, overrides, scenes=scenes)
    if cache_hit:
        n_scenes = 0
    elif scenes is not None:
        n_scenes = len(scenes)
    elif n_full_scenes is not None:
        n_scenes = int(n_full_scenes)
    else:
        n_scenes = len(dataset_builder(cfg).scenes)

    t0 = time.perf_counter()
    score = evaluate(
        cfg,
        overrides,
        scenes=scenes,
        optuna_data=optuna_data,
        dataset_builder=dataset_builder,
    )
    return score, n_scenes, time.perf_counter() - t0


def bootstrap_detection_cache(
    cfg: GlobalConfig,
    dataset_builder: DatasetBuilder = default_dataset_builder,
) -> None:
    """Populate the detection cache once if ``cache_path`` is set but missing.

    Without a populated cache, every trial would recompute detections from
    scratch — a serious performance hit for coordinate search. We trigger one
    full-dataset inference pass so subsequent trials all hit the cache.

    No-op if ``cache_path`` is None (logs the same warning as the TPE path)
    or if the cache already exists.
    """
    cache_path = cfg.object_detection.cache_path
    if cache_path is None:
        logger.warning(
            'Detection caching is disabled (object_detection.cache_path is None). '
            'Each trial will recompute detections from scratch.'
        )
        return

    if os.path.exists(cache_path):
        return

    logger.info(
        f'Detection cache "{cache_path}" not found. Running a one-shot full-dataset '
        f'inference to populate it before the optimization sweep starts.'
    )
    bootstrap_cfg = cfg.override({})
    bootstrap_cfg.inference.override = True
    run_inference(bootstrap_cfg, dataset_builder=dataset_builder)


def guard_optimization_dir(cfg: GlobalConfig, study_name: str) -> str:
    """Refuse to start if the optimization output directory already exists.

    Returns the resolved split-level path so the caller can persist results.
    """
    split_path = conventions.get_split_results_path(
        master_path=cfg.path.master,
        dataset_type=cfg.dataset.type,
        experiment_name=cfg.experiment,
        split=cfg.inference.split,
        dataset_name=cfg.dataset.name,
    )
    optimization_dir = conventions.get_optimization_path(split_path, study_name)
    if os.path.exists(optimization_dir):
        raise FileExistsError(
            f'Optimization "{study_name}" already exists at "{optimization_dir}". '
            f'Choose a different study_name to avoid overwriting previous results.'
        )
    return split_path


def log_trial_to_mlflow(
    cfg: GlobalConfig,
    optuna_info: OptunaOutputData,
    *,
    extra_metrics: Optional[Dict[str, float]] = None,
) -> None:
    """Log a trial to MLflow (no-op if the integration is disabled).

    ``extra_metrics`` is merged with the eval/FPS metrics so callers can attach
    optimizer-specific values (e.g. ``scenes_evaluated``, ``trial_wall_time_s``)
    without needing to know the MLflow plumbing.
    """
    from motrack.tools.mlflow_logger import load_and_log_run
    load_and_log_run(cfg, optuna_info=optuna_info, extra_metrics=extra_metrics)
