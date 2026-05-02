"""
Optuna-driven optimization (random / TPE / warm-started TPE).

Per-trial objective uses ``common.evaluate`` so caching, dataset construction,
and the score-extraction contract stay consistent across all algorithms.
"""
import logging
from typing import Any, Callable, Dict

import numpy as np
import optuna

from motrack.common import conventions
from motrack.config_parser import GlobalConfig, SearchSpaceParam
from motrack.tools.dataset_builder import DatasetBuilder, default_dataset_builder
from motrack.tools.inference import OptunaOutputData
from motrack.tools.optimization.common import (
    evaluate_with_metrics,
    extract_base_params,
    guard_optimization_dir,
    log_trial_to_mlflow,
)
from motrack.tools.optimization.results import OptimizationResults, TrialResult

logger = logging.getLogger('Tool-Optimize')


def create_sampler(name: str, sampler_params: dict) -> optuna.samplers.BaseSampler:
    """Create an Optuna sampler by name."""
    if name == 'random':
        return optuna.samplers.RandomSampler()
    if name in ('tpe', 'warm_tpe'):
        params = dict(sampler_params)
        if isinstance(params.get('gamma'), float):
            gamma_ratio = params['gamma']
            params['gamma'] = lambda x, _r=gamma_ratio: max(1, int(np.ceil(_r * x)))
        return optuna.samplers.TPESampler(**params)
    raise ValueError(f'Unknown sampler: {name}')


def sample_params(trial: optuna.Trial, search_space: Dict[str, SearchSpaceParam]) -> Dict[str, Any]:
    """Sample parameters from the search space using an Optuna trial.

    Parameters with ``min_param`` are sampled after the param they depend on,
    using the already-sampled value as their effective lower bound.
    """
    independent = {k: v for k, v in search_space.items() if v.min_param is None and v.max_param is None}
    dependent = {k: v for k, v in search_space.items() if v.min_param is not None or v.max_param is not None}

    params: Dict[str, Any] = {}

    def _sample(dotpath: str, spec: SearchSpaceParam) -> Any:
        low = spec.low
        high = spec.high
        if spec.min_param is not None:
            assert spec.min_param in params, (
                f'min_param "{spec.min_param}" for "{dotpath}" must appear earlier in search_space'
            )
            low = max(low, params[spec.min_param]) if low is not None else params[spec.min_param]
        if spec.max_param is not None:
            assert spec.max_param in params, (
                f'max_param "{spec.max_param}" for "{dotpath}" must appear earlier in search_space'
            )
            high = min(high, params[spec.max_param]) if high is not None else params[spec.max_param]
        if spec.type == 'int':
            return trial.suggest_int(dotpath, int(low), int(high), step=int(spec.step) if spec.step is not None else 1)
        if spec.type == 'float':
            return trial.suggest_float(dotpath, low, high, step=spec.step, log=spec.log)
        if spec.type == 'categorical':
            return trial.suggest_categorical(dotpath, spec.choices)
        raise ValueError(f'Unknown search space param type: {spec.type}')

    for dotpath, spec in independent.items():
        params[dotpath] = _sample(dotpath, spec)
    for dotpath, spec in dependent.items():
        params[dotpath] = _sample(dotpath, spec)

    return params


def create_study(cfg: GlobalConfig) -> optuna.Study:
    """Create and configure an Optuna study from the optimizer config.

    For ``warm_tpe``, the base config parameter values are enqueued as the
    first trial so that the TPE sampler can use them as a starting point.
    """
    optim_cfg = cfg.optimizer
    search_space = optim_cfg.search_space

    sampler = create_sampler(optim_cfg.sampler, optim_cfg.sampler_params)
    study = optuna.create_study(
        study_name=optim_cfg.study_name,
        sampler=sampler,
        direction=optim_cfg.direction,
    )

    if optim_cfg.sampler == 'warm_tpe':
        study.enqueue_trial(extract_base_params(cfg, search_space))

    return study


def create_objective(
    cfg: GlobalConfig,
    search_space: Dict[str, SearchSpaceParam],
    dataset_builder: DatasetBuilder = default_dataset_builder,
) -> Callable[[optuna.Trial], float]:
    """Build the Optuna objective function for HOTA maximization."""
    optim_cfg = cfg.optimizer
    n_full_scenes = len(dataset_builder(cfg).scenes)

    def objective(trial: optuna.Trial) -> float:
        params = sample_params(trial, search_space)
        trial_cfg = cfg.override(params)
        config_hash = trial_cfg.hash
        trial.set_user_attr('config_hash', config_hash)

        optuna_data = OptunaOutputData(
            study_name=optim_cfg.study_name,
            trial_number=trial.number,
            trial_params=params,
        )

        hota, scenes_evaluated, wall_time_s = evaluate_with_metrics(
            cfg,
            params,
            optuna_data=optuna_data,
            dataset_builder=dataset_builder,
            n_full_scenes=n_full_scenes,
        )
        trial.set_user_attr('scenes_evaluated', scenes_evaluated)
        trial.set_user_attr('wall_time_s', wall_time_s)
        logger.info(
            f'Trial {trial.number}: HOTA={hota:.4f}, '
            f'scenes={scenes_evaluated}, wall_time={wall_time_s:.1f}s, params={params}'
        )
        log_trial_to_mlflow(
            trial_cfg,
            optuna_data,
            extra_metrics={
                'scenes_evaluated': float(scenes_evaluated),
                'trial_wall_time_s': float(wall_time_s),
            },
        )
        return hota

    return objective


def save_optimization_results(cfg: GlobalConfig, study: optuna.Study) -> None:
    """Save best trial + all-trials summary under ``optimizations/{study_name}/``."""
    best = study.best_trial
    logger.info(f'Best trial #{best.number}: HOTA={best.value:.4f}, params={best.params}')

    def _trial_result(t: optuna.trial.FrozenTrial) -> TrialResult:
        return TrialResult(
            number=t.number,
            value=t.value,
            params=t.params,
            state=t.state.name,
            config_hash=t.user_attrs['config_hash'],
            scenes_evaluated=int(t.user_attrs.get('scenes_evaluated', 0)),
            wall_time_s=float(t.user_attrs.get('wall_time_s', 0.0)),
        )

    results = OptimizationResults(
        study_name=study.study_name,
        algorithm=cfg.optimizer.sampler,
        best_trial=_trial_result(best),
        all_trials=[_trial_result(t) for t in study.trials],
    )

    split_path = conventions.get_split_results_path(
        master_path=cfg.path.master,
        dataset_type=cfg.dataset.type,
        experiment_name=cfg.experiment,
        split=cfg.inference.split,
        dataset_name=cfg.dataset.name,
    )
    results_path = conventions.get_optimization_results_path(split_path, study.study_name)
    results.save(results_path)
    logger.info(f'Optimization results saved to "{results_path}".')


def run_tpe(
    cfg: GlobalConfig,
    dataset_builder: DatasetBuilder = default_dataset_builder,
) -> None:
    """Optuna-driven optimization entry point (random / TPE / warm_tpe)."""
    optim_cfg = cfg.optimizer
    search_space = optim_cfg.search_space

    guard_optimization_dir(cfg, optim_cfg.study_name)

    if cfg.object_detection.cache_path is None:
        logger.warning(
            'Detection caching is disabled (object_detection.cache_path is None). '
            'Each trial will recompute detections from scratch.'
        )

    cfg.inference.override = True

    study = create_study(cfg)
    objective = create_objective(cfg, search_space, dataset_builder=dataset_builder)
    study.optimize(objective, n_trials=optim_cfg.n_trials)
    save_optimization_results(cfg, study)
