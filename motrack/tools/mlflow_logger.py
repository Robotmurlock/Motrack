"""
Optional MLflow logging adapter for Motrack pipelines.

When mlflow is installed and tracking is enabled, logs params, metrics,
tags, and artifacts to the configured MLflow backend.  When mlflow is
not installed, all operations are silent no-ops.

Toggle via the ``mlflow.enabled`` config flag.
"""
import json
import logging
import os
from typing import Any, Dict, Optional

from motrack.common import conventions
from motrack.config_parser import GlobalConfig, MlflowConfig
from motrack.eval.results import EvalResults
from motrack.tools.inference import OptunaOutputData

logger = logging.getLogger('MlflowLogger')

try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None  # type: ignore[assignment]
    _MLFLOW_AVAILABLE = False


def mlflow_available() -> bool:
    """Return True if the mlflow package is installed."""
    return _MLFLOW_AVAILABLE

_KEY_METRICS: Dict[tuple, str] = {
    ('HOTA', 'HOTA'): 'HOTA',
    ('HOTA', 'DetA'): 'DetA',
    ('HOTA', 'AssA'): 'AssA',
    ('HOTA', 'DetRe'): 'DetRe',
    ('HOTA', 'DetPr'): 'DetPr',
    ('HOTA', 'AssRe'): 'AssRe',
    ('HOTA', 'AssPr'): 'AssPr',
    ('HOTA', 'LocA'): 'LocA',
    ('CLEAR', 'MOTA'): 'MOTA',
    ('CLEAR', 'MOTP'): 'MOTP',
    ('CLEAR', 'IDSW'): 'IDSW',
    ('CLEAR', 'CLR_Re'): 'CLR_Re',
    ('CLEAR', 'CLR_Pr'): 'CLR_Pr',
    ('CLEAR', 'FP_per_frame'): 'FP_per_frame',
    ('Identity', 'IDF1'): 'IDF1',
    ('Identity', 'IDR'): 'IDR',
    ('Identity', 'IDP'): 'IDP',
    ('Count', 'Dets'): 'Dets',
    ('Count', 'GT_Dets'): 'GT_Dets',
    ('Count', 'IDs'): 'IDs',
    ('Count', 'GT_IDs'): 'GT_IDs',
}


def is_mlflow_enabled(mlflow_cfg: MlflowConfig) -> bool:
    """Check if MLflow is available and enabled in the config."""
    if not _MLFLOW_AVAILABLE:
        logger.warning('MLflow is not installed. Skipping MLflow logging.')
        return False
    return mlflow_cfg.enabled


def log_run(
    cfg: GlobalConfig,
    eval_results: EvalResults,
    fps_stats: Optional[Dict[str, Any]] = None,
    optuna_info: Optional[OptunaOutputData] = None,
    *,
    extra_metrics: Optional[Dict[str, float]] = None,
) -> None:
    """Log a completed tracker run to MLflow.

    Creates/reuses an MLflow experiment and logs one run with params,
    metrics, tags, and artifacts.  Safe to call when mlflow is not
    installed (silently returns).

    Args:
        cfg: The GlobalConfig used for this run.
        eval_results: Evaluation results to log as metrics.
        fps_stats: Optional FPS statistics dict.
        optuna_info: Optional Optuna trial metadata.
        extra_metrics: Optional optimizer-specific metrics (e.g.
            ``scenes_evaluated``, ``trial_wall_time_s``) merged into the
            metrics block.
    """
    if not is_mlflow_enabled(cfg.mlflow):
        return

    if cfg.mlflow.tracking_uri is not None:
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

    dataset_name = conventions.get_dataset_name(cfg.dataset.type, cfg.dataset.name)
    experiment_name = f'{dataset_name}/{cfg.experiment}/{cfg.inference.split}'
    config_hash = cfg.hash

    if _find_existing_run(experiment_name, config_hash) is not None:
        logger.debug(f'MLflow run for config_hash={config_hash} already exists. Skipping.')
        return

    mlflow.set_experiment(experiment_name)

    run_name = config_hash
    tags: Dict[str, str] = {
        'config_hash': config_hash,
        'dataset.type': cfg.dataset.type,
        'experiment': cfg.experiment,
        'split': cfg.inference.split,
        'algorithm.name': cfg.algorithm.name,
    }
    if optuna_info is not None:
        tags['optuna.study_name'] = optuna_info.study_name
        tags['optuna.trial_number'] = str(optuna_info.trial_number)
        run_name = f'{config_hash} (trial-{optuna_info.trial_number})'

    with mlflow.start_run(run_name=run_name, tags=tags):
        mlflow.log_params(_flatten_params(cfg))

        metrics = _extract_metrics(eval_results)
        if fps_stats is not None and 'e2e_fps' in fps_stats:
            metrics['e2e_fps'] = float(fps_stats['e2e_fps'])
        if extra_metrics:
            for k, v in extra_metrics.items():
                metrics[k] = float(v)
        mlflow.log_metrics(metrics)

        config_path = conventions.get_config_snapshot_path(cfg.experiment_path)
        eval_path = conventions.get_eval_results_path(cfg.experiment_path)
        for path in (config_path, eval_path):
            if os.path.exists(path):
                mlflow.log_artifact(path)

    logger.info(
        f'Logged MLflow run: experiment="{experiment_name}", '
        f'run="{run_name}", HOTA={metrics.get("HOTA", "N/A")}'
    )


def load_and_log_run(
    cfg: GlobalConfig,
    optuna_info: Optional[OptunaOutputData] = None,
    *,
    extra_metrics: Optional[Dict[str, float]] = None,
) -> None:
    """Load eval results and FPS stats from disk and log to MLflow.

    Convenience wrapper used by tool entry points to avoid duplicating
    the file-loading boilerplate.

    Args:
        cfg: The GlobalConfig used for this run.
        optuna_info: Optional Optuna trial metadata.
        extra_metrics: Optional optimizer-specific metrics passed through to
            :func:`log_run` (e.g. ``scenes_evaluated``, ``trial_wall_time_s``).
    """
    if not is_mlflow_enabled(cfg.mlflow):
        return

    eval_path = conventions.get_eval_results_path(cfg.experiment_path)
    if not os.path.exists(eval_path):
        return
    eval_results = EvalResults.load(eval_path)

    fps_stats = None
    fps_path = conventions.get_fps_stats_path(cfg.experiment_path)
    if os.path.exists(fps_path):
        with open(fps_path, 'r', encoding='utf-8') as f:
            fps_stats = json.load(f)

    log_run(
        cfg,
        eval_results,
        fps_stats=fps_stats,
        optuna_info=optuna_info,
        extra_metrics=extra_metrics,
    )


def _find_existing_run(experiment_name: str, config_hash: str) -> Optional[str]:
    """Return run_id of an existing run with the given config_hash, or None."""
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            return None
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.config_hash = '{config_hash}'",
            max_results=1,
        )
        if len(runs) > 0:
            return runs.iloc[0]['run_id']
    except Exception as e:
        logger.warning(f'Failed to search MLflow runs: {e}')
    return None


def _flatten_params(cfg: GlobalConfig) -> Dict[str, str]:
    """Extract key config params as flat dotpath strings for MLflow."""
    params: Dict[str, str] = {
        'algorithm.name': cfg.algorithm.name,
        'object_detection.type': cfg.object_detection.type,
        'dataset.type': cfg.dataset.type,
        'inference.split': cfg.inference.split,
        'inference.clip': str(cfg.inference.clip),
    }
    _flatten_dict(cfg.algorithm.params, 'algorithm.params', params)
    return params


def _flatten_dict(d: Any, prefix: str, out: Dict[str, str]) -> None:
    """Recursively flatten a nested dict into dotpath keys."""
    if isinstance(d, dict):
        for k, v in d.items():
            _flatten_dict(v, f'{prefix}.{k}', out)
    else:
        out[prefix] = str(d)


def _extract_metrics(eval_results: EvalResults) -> Dict[str, float]:
    """Extract key numeric metrics from combined eval results."""
    metrics: Dict[str, float] = {}
    for (group, field_name), mlflow_name in _KEY_METRICS.items():
        value = eval_results.combined.get(group, {}).get(field_name)
        if value is None:
            continue
        if isinstance(value, (int, float)):
            metrics[mlflow_name] = float(value)
    return metrics
