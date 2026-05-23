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
from motrack.optimization.optuna.params import (
    GPParams,
    HyperbandPrunerConfig,
    RandomParams,
    TPEParams,
    WarmGPParams,
    WarmTPEParams,
)
from motrack.optimization.results import OptimizationResults, TrialResult

logger = logging.getLogger('Tool-Optimize')

OptunaParams = Union[RandomParams, TPEParams, WarmTPEParams, GPParams, WarmGPParams]


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
        if self._sampler_name in ('gp', 'warm_gp'):
            return optuna.samplers.GPSampler(
                n_startup_trials=self._params.n_startup_trials,
                seed=self._params.seed,
                deterministic_objective=self._params.deterministic_objective,
            )
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

    def _pruner_config(self) -> 'HyperbandPrunerConfig | None':
        """Coerce ``params.pruner`` (a dict from YAML) into a typed config."""
        raw = getattr(self._params, 'pruner', None)
        if raw is None:
            return None
        return HyperbandPrunerConfig(**raw)

    def _build_pruner(self) -> 'optuna.pruners.BasePruner | None':
        cfg = self._pruner_config()
        if cfg is None:
            return None
        return optuna.pruners.HyperbandPruner(
            min_resource=cfg.min_resource,
            max_resource=cfg.max_resource,
            reduction_factor=cfg.reduction_factor,
        )

    def _build_study(self) -> optuna.Study:
        optim_cfg = self._cfg.optimizer
        study_kwargs: Dict[str, Any] = dict(
            study_name=optim_cfg.study_name,
            sampler=self._build_sampler(),
            direction=optim_cfg.direction,
        )
        pruner = self._build_pruner()
        if pruner is not None:
            study_kwargs['pruner'] = pruner
        study = optuna.create_study(**study_kwargs)
        if self._sampler_name in ('warm_tpe', 'warm_gp'):
            study.enqueue_trial(extract_base_params(self._cfg, optim_cfg.search_space))
        return study

    @staticmethod
    def _rung_schedule(cfg: HyperbandPrunerConfig) -> 'list[int]':
        """Compute the per-step scene-count schedule.

        Rung k uses ``min_resource * reduction_factor**k`` scenes, capped
        at ``max_resource``. The final element is always ``max_resource``.
        """
        rungs: list[int] = []
        n = cfg.min_resource
        while n < cfg.max_resource:
            rungs.append(n)
            n *= cfg.reduction_factor
        rungs.append(cfg.max_resource)
        return rungs

    # --- Objective ---------------------------------------------------------

    def _build_objective(self) -> Callable[[optuna.Trial], float]:
        cfg = self._cfg
        optim_cfg = cfg.optimizer
        search_space = optim_cfg.search_space
        dataset_builder = self._dataset_builder
        full_scenes = list(dataset_builder(cfg).scenes)
        n_full_scenes = len(full_scenes)

        pruner_cfg = self._pruner_config()
        if pruner_cfg is None:
            return self._build_single_fidelity_objective(n_full_scenes)
        return self._build_pruned_objective(pruner_cfg, full_scenes)

    def _build_single_fidelity_objective(self, n_full_scenes: int) -> Callable[[optuna.Trial], float]:
        cfg = self._cfg
        optim_cfg = cfg.optimizer
        search_space = optim_cfg.search_space
        dataset_builder = self._dataset_builder

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

    def _build_pruned_objective(
        self,
        pruner_cfg: HyperbandPrunerConfig,
        full_scenes: 'list[str]',
    ) -> Callable[[optuna.Trial], float]:
        cfg = self._cfg
        optim_cfg = cfg.optimizer
        search_space = optim_cfg.search_space
        dataset_builder = self._dataset_builder
        n_full_scenes = len(full_scenes)
        rungs = self._rung_schedule(pruner_cfg)
        seed = pruner_cfg.seed

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

            # Per-trial RNG so each trial draws its own subsets at each rung.
            rng = np.random.default_rng(None if seed is None else seed + trial.number)
            cum_scenes = 0
            cum_wall = 0.0
            last_hota = 0.0
            for step, n_scenes in enumerate(rungs):
                if n_scenes >= n_full_scenes:
                    scene_subset = None
                else:
                    idx = rng.choice(n_full_scenes, size=n_scenes, replace=False)
                    scene_subset = [full_scenes[int(i)] for i in idx]
                hota, n_used, wall = evaluate_with_metrics(
                    cfg,
                    params,
                    scenes=scene_subset,
                    optuna_data=optuna_data,
                    dataset_builder=dataset_builder,
                    n_full_scenes=n_full_scenes,
                )
                cum_scenes += n_used
                cum_wall += wall
                last_hota = hota
                trial.report(hota, step=step)
                if step < len(rungs) - 1 and trial.should_prune():
                    trial.set_user_attr('config_hash', config_hash)
                    trial.set_user_attr('scenes_evaluated', cum_scenes)
                    trial.set_user_attr('wall_time_s', cum_wall)
                    logger.info(
                        f'Trial {trial.number} PRUNED at step {step} '
                        f'(rung n={n_scenes}): HOTA={hota:.4f}, scenes={cum_scenes}'
                    )
                    log_trial_to_mlflow(
                        trial_cfg,
                        optuna_data,
                        extra_metrics={
                            'scenes_evaluated': float(cum_scenes),
                            'trial_wall_time_s': float(cum_wall),
                        },
                    )
                    raise optuna.TrialPruned()

            trial.set_user_attr('scenes_evaluated', cum_scenes)
            trial.set_user_attr('wall_time_s', cum_wall)
            logger.info(
                f'Trial {trial.number} (full): HOTA={last_hota:.4f}, '
                f'scenes={cum_scenes}, wall={cum_wall:.1f}s'
            )
            log_trial_to_mlflow(
                trial_cfg,
                optuna_data,
                extra_metrics={
                    'scenes_evaluated': float(cum_scenes),
                    'trial_wall_time_s': float(cum_wall),
                },
            )
            return last_hota

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
