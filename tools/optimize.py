"""
Tracker hyperparameter optimization using Optuna.
"""
import logging
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
from tools.eval import _run_eval_inner
from tools.inference import _run_inference_inner

logger = logging.getLogger('Tool-Optimize')


def create_sampler(name: str) -> optuna.samplers.BaseSampler:
    """Create an Optuna sampler by name."""
    if name == 'random':
        return optuna.samplers.RandomSampler()
    if name in ('tpe', 'warm_tpe'):
        return optuna.samplers.TPESampler()
    raise ValueError(f'Unknown sampler: {name}')


def sample_params(trial: optuna.Trial, search_space: Dict[str, SearchSpaceParam]) -> Dict[str, Any]:
    """Sample parameters from search space using an Optuna trial."""
    params: Dict[str, Any] = {}
    for dotpath, spec in search_space.items():
        if spec.type == 'int':
            params[dotpath] = trial.suggest_int(
                dotpath, int(spec.low), int(spec.high),
                step=int(spec.step) if spec.step is not None else 1,
            )
        elif spec.type == 'float':
            params[dotpath] = trial.suggest_float(
                dotpath, spec.low, spec.high,
                step=spec.step, log=spec.log,
            )
        elif spec.type == 'categorical':
            params[dotpath] = trial.suggest_categorical(dotpath, spec.choices)
        else:
            raise ValueError(f'Unknown search space param type: {spec.type}')
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

    sampler = create_sampler(optim_cfg.sampler)
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
    thresholds.

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

        inference_output = InferenceOutputData(
            created_at=datetime.now().isoformat(),
            optuna=OptunaOutputData(
                study_name=optim_cfg.study_name,
                trial_number=trial.number,
                trial_params=params,
            ),
        )

        _run_inference_inner(trial_cfg, inference_output=inference_output)
        results = _run_eval_inner(trial_cfg)

        hota = float(np.mean(results['combined']['HOTA']['HOTA']))
        logger.info(f'Trial {trial.number}: HOTA={hota:.4f}, params={params}')
        return hota

    return objective


def save_optimization_results(cfg: GlobalConfig, study: optuna.Study) -> None:
    """
    Save the optimization results to a JSON file at the split level.

    Writes the best trial (number, HOTA value, params) and a summary of all
    trials to ``optimization_results.json`` under the split results directory.

    Args:
        cfg: Global config used during the optimization run.
        study: Completed Optuna study.
    """
    best = study.best_trial
    logger.info(f'Best trial #{best.number}: HOTA={best.value:.4f}, params={best.params}')

    results = OptimizationResults(
        study_name=study.study_name,
        best_trial=TrialResult(
            number=best.number,
            value=best.value,
            params=best.params,
            state=best.state.name,
        ),
        all_trials=[
            TrialResult(
                number=t.number,
                value=t.value,
                params=t.params,
                state=t.state.name,
            )
            for t in study.trials
        ],
    )

    split_path = conventions.get_split_results_path(
        master_path=cfg.path.master,
        dataset_type=cfg.dataset.type,
        experiment_name=cfg.experiment,
        split=cfg.inference.split,
        dataset_name=cfg.dataset.name,
    )
    results_path = conventions.get_optimization_results_path(split_path)
    results.save(results_path)
    logger.info(f'Optimization results saved to "{results_path}".')


def _run_optimize_inner(cfg: GlobalConfig) -> None:
    assert cfg.optimizer is not None, 'optimizer config is required'
    optim_cfg = cfg.optimizer
    search_space = optim_cfg.search_space

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


@hydra.main(config_path=DANCETRACK_TRACKERS_CONFIG_PATH, config_name='sort_optimize', version_base='1.1')
@pipeline.task('optimize')
def main(cfg: GlobalConfig) -> None:
    _run_optimize_inner(cfg)


if __name__ == '__main__':
    main()
