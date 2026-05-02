"""
Single Optuna wrapper covering the random / TPE / warm_tpe samplers.

One class, three sampler-name strings — adding a new Optuna sampler later
needs only a new params dataclass + a registry line, not a new wrapper
class. The sampler-name string is read off ``cfg.optimizer.sampler``.
"""
import logging
from typing import Any, Callable, Dict, Union

import numpy as np
import optuna

from motrack.common import conventions
from motrack.config_parser import GlobalConfig, SearchSpaceParam
from motrack.tools.dataset_builder import DatasetBuilder
from motrack.tools.inference import OptunaOutputData
from motrack.optimization.base import OptimizationPipeline
from motrack.optimization.common import (
    evaluate_with_metrics,
    extract_base_params,
    guard_optimization_dir,
    log_trial_to_mlflow,
)
from motrack.optimization.optuna.params import RandomParams, TPEParams, WarmTPEParams
from motrack.optimization.results import OptimizationResults, TrialResult

logger = logging.getLogger('Tool-Optimize')

OptunaParams = Union[RandomParams, TPEParams, WarmTPEParams]


def _sample_params(
    trial: optuna.Trial,
    search_space: Dict[str, SearchSpaceParam],
) -> Dict[str, Any]:
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


class OptunaPipeline(OptimizationPipeline):
    """Drives one Optuna study (random / TPE / warm_tpe).

    The choice of Optuna sampler class is decided by ``cfg.optimizer.sampler``;
    the typed ``params`` payload supplies the kwargs.
    """

    def __init__(
        self,
        cfg: GlobalConfig,
        dataset_builder: DatasetBuilder,
        params: OptunaParams,
    ) -> None:
        super().__init__(cfg, dataset_builder, params)
        self._sampler_name: str = cfg.optimizer.sampler

    # --- Sampler / study construction --------------------------------------

    def _build_sampler(self) -> optuna.samplers.BaseSampler:
        if self._sampler_name == 'random':
            return optuna.samplers.RandomSampler(seed=self._params.seed)
        kwargs = {
            'multivariate': self._params.multivariate,
            'n_startup_trials': self._params.n_startup_trials,
            'n_ei_candidates': self._params.n_ei_candidates,
            'seed': self._params.seed,
        }
        if self._params.gamma is not None:
            ratio = float(self._params.gamma)
            kwargs['gamma'] = lambda x, _r=ratio: max(1, int(np.ceil(_r * x)))
        return optuna.samplers.TPESampler(**kwargs)

    def _build_study(self) -> optuna.Study:
        optim_cfg = self._cfg.optimizer
        study = optuna.create_study(
            study_name=optim_cfg.study_name,
            sampler=self._build_sampler(),
            direction=optim_cfg.direction,
        )
        if self._sampler_name == 'warm_tpe':
            study.enqueue_trial(extract_base_params(self._cfg, optim_cfg.search_space))
        return study

    # --- Objective ---------------------------------------------------------

    def _build_objective(self) -> Callable[[optuna.Trial], float]:
        cfg = self._cfg
        optim_cfg = cfg.optimizer
        search_space = optim_cfg.search_space
        dataset_builder = self._dataset_builder
        n_full_scenes = len(dataset_builder(cfg).scenes)

        def objective(trial: optuna.Trial) -> float:
            params = _sample_params(trial, search_space)
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

    # --- Save --------------------------------------------------------------

    def _save_results(self, study: optuna.Study) -> None:
        cfg = self._cfg
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
            algorithm=self._sampler_name,
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

    # --- Public entry point ------------------------------------------------

    def run(self) -> None:
        cfg = self._cfg
        optim_cfg = cfg.optimizer

        guard_optimization_dir(cfg, optim_cfg.study_name)

        if cfg.object_detection.cache_path is None:
            logger.warning(
                'Detection caching is disabled (object_detection.cache_path is None). '
                'Each trial will recompute detections from scratch.'
            )

        cfg.inference.override = True

        study = self._build_study()
        study.optimize(self._build_objective(), n_trials=optim_cfg.n_trials)
        self._save_results(study)
