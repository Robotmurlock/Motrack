"""
Tracker hyperparameter optimization using Optuna.
"""
import logging
import os
from datetime import datetime
from typing import Any, Callable, Dict

import hydra
import numpy as np
import optuna
from motrack.common import conventions
from motrack.common.project import DANCETRACK_TRACKERS_CONFIG_PATH
from motrack.config_parser import GlobalConfig, SearchSpaceParam
from motrack.utils import pipeline
from tools.data import OptimizationResults, OptunaOutputData, InferenceOutputData, TrialResult
from tools.data.eval import EvalResults
from tools.eval import _run_eval_inner
from tools.inference import _run_inference_inner

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
    """Sample parameters from search space using an Optuna trial.

    Parameters with ``min_param`` are sampled after the param they depend on,
    using the already-sampled value as their effective lower bound.
    """
    # Split into independent and dependent params to ensure dependencies are resolved first
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


def extract_base_params(cfg: GlobalConfig, search_space: Dict[str, SearchSpaceParam]) -> Dict[str, Any]:
    """Extract current config values for search space params (for warm-start)."""
    return {dotpath: cfg.resolve(dotpath) for dotpath in search_space}


def create_study(cfg: GlobalConfig) -> optuna.Study:
    """
    Create and configure an Optuna study from the optimizer config.

    Instantiates the study with the configured sampler (random, TPE, or warm-started TPE).
    For ``warm_tpe``, the base config parameter values are enqueued as the first trial
    so that the TPE sampler can use them as a starting point.

    Args:
        cfg: Global config with a populated ``optimizer`` field.

    Returns:
        Configured Optuna study ready for optimization.
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
) -> Callable[[optuna.Trial], float]:
    """
    Build the Optuna objective function for HOTA maximization.

    Each call to the returned objective runs a full inference + evaluation cycle
    with sampled hyperparameters and returns the mean HOTA score across alpha
    thresholds.  If a trial's config hash already has evaluation results, the
    existing HOTA is returned without re-running inference or evaluation.

    Args:
        cfg: Base global config (deep-copied per trial via ``cfg.override``).
        search_space: Parsed search space parameters.

    Returns:
        Objective callable suitable for ``study.optimize(objective, ...)``.
    """
    optim_cfg = cfg.optimizer

    def objective(trial: optuna.Trial) -> float:
        params = sample_params(trial, search_space)
        trial_cfg = cfg.override(params)
        trial_cfg.inference.override = True

        config_hash = trial_cfg.hash
        trial.set_user_attr('config_hash', config_hash)

        optuna_data = OptunaOutputData(
            study_name=optim_cfg.study_name,
            trial_number=trial.number,
            trial_params=params,
        )

        eval_path = conventions.get_eval_results_path(trial_cfg.experiment_path)
        if os.path.exists(eval_path):
            eval_results = EvalResults.load(eval_path)
            hota = float(np.mean(eval_results.combined['HOTA']['HOTA']))
            logger.info(f'Trial {trial.number}: HOTA={hota:.4f} (cached, hash={config_hash}), params={params}')
            _log_trial_to_mlflow(trial_cfg, optuna_data)
            return hota

        inference_output = InferenceOutputData(
            created_at=datetime.now().isoformat(),
            optuna=optuna_data,
        )

        _run_inference_inner(trial_cfg, inference_output=inference_output)
        results = _run_eval_inner(trial_cfg)

        hota = float(np.mean(results['combined']['HOTA']['HOTA']))
        logger.info(f'Trial {trial.number}: HOTA={hota:.4f}, params={params}')
        _log_trial_to_mlflow(trial_cfg, optuna_data)
        return hota

    return objective


def _log_trial_to_mlflow(cfg: GlobalConfig, optuna_info: OptunaOutputData) -> None:
    """Log an optimization trial to MLflow (no-op if mlflow not installed)."""
    from motrack.tools.mlflow_logger import load_and_log_run
    load_and_log_run(cfg, optuna_info=optuna_info)


def save_optimization_results(cfg: GlobalConfig, study: optuna.Study) -> None:
    """
    Save the optimization results under ``optimizations/{study_name}/``.

    Writes the best trial (number, HOTA value, params, config_hash) and a
    summary of all trials to ``optimization_results.json``.

    Args:
        cfg: Global config used during the optimization run.
        study: Completed Optuna study.
    """
    best = study.best_trial
    logger.info(f'Best trial #{best.number}: HOTA={best.value:.4f}, params={best.params}')

    def _trial_result(t: optuna.trial.FrozenTrial) -> TrialResult:
        return TrialResult(
            number=t.number,
            value=t.value,
            params=t.params,
            state=t.state.name,
            config_hash=t.user_attrs['config_hash'],
        )

    results = OptimizationResults(
        study_name=study.study_name,
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


def _run_optimize_inner(cfg: GlobalConfig) -> None:
    assert cfg.optimizer is not None, 'optimizer config is required'
    optim_cfg = cfg.optimizer
    search_space = optim_cfg.search_space

    split_path = conventions.get_split_results_path(
        master_path=cfg.path.master,
        dataset_type=cfg.dataset.type,
        experiment_name=cfg.experiment,
        split=cfg.inference.split,
        dataset_name=cfg.dataset.name,
    )
    optimization_dir = conventions.get_optimization_path(split_path, optim_cfg.study_name)
    if os.path.exists(optimization_dir):
        raise FileExistsError(
            f'Optimization "{optim_cfg.study_name}" already exists at "{optimization_dir}". '
            f'Choose a different study_name to avoid overwriting previous results.'
        )

    if cfg.object_detection.cache_path is None:
        logger.warning(
            'Detection caching is disabled (object_detection.cache_path is None). '
            'Each trial will recompute detections from scratch.'
        )

    cfg.inference.override = True

    study = create_study(cfg)
    objective = create_objective(cfg, search_space)
    study.optimize(objective, n_trials=optim_cfg.n_trials)
    save_optimization_results(cfg, study)


@hydra.main(config_path=DANCETRACK_TRACKERS_CONFIG_PATH, config_name='optimize_sort', version_base='1.1')
@pipeline.task('optimize')
def main(cfg: GlobalConfig) -> None:
    _run_optimize_inner(cfg)


if __name__ == '__main__':
    main()
