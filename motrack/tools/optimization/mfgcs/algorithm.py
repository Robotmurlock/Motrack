"""
Multi-Fidelity Greedy Coordinate Search (MFGCS) algorithm.

Outer loop sweeps parameters, optimizes each via a low-fidelity scene subset,
validates the candidate on the full split, and accepts only strict
improvements. The sweep ends once a full pass produces no accepted move.
"""
import logging
import math
from typing import Any, Dict, List, Optional

from motrack.common import conventions
from motrack.config_parser import GlobalConfig, SearchSpaceParam
from motrack.tools.dataset_builder import DatasetBuilder, default_dataset_builder
from motrack.tools.inference import OptunaOutputData
from motrack.tools.optimization.common import (
    bootstrap_detection_cache,
    evaluate,
    extract_base_params,
    guard_optimization_dir,
    log_trial_to_mlflow,
)
from motrack.tools.optimization.mfgcs.coordinate import (
    SearchWindow,
    coordinate_optimizer_factory,
)
from motrack.tools.optimization.mfgcs.results import (
    MFGCSCoordinateRecord,
    MFGCSSweepRecord,
)
from motrack.tools.optimization.mfgcs.scene_sampler import scene_sampler_factory
from motrack.tools.optimization.mfgcs.shrinking import SearchSpaceShrinker
from motrack.tools.optimization.results import OptimizationResults, TrialResult

logger = logging.getLogger('Tool-Optimize-MFGCS')


def _values_equal(a: Any, b: Any) -> bool:
    """Equality with FP tolerance for floats; strict equality otherwise."""
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
        except (TypeError, ValueError):
            return a == b
    return a == b


class MFGCSAlgorithm:
    """Driver for one MFGCS optimization run."""

    def __init__(
        self,
        cfg: GlobalConfig,
        dataset_builder: DatasetBuilder = default_dataset_builder,
    ) -> None:
        assert cfg.optimizer is not None, 'optimizer config is required'
        assert cfg.optimizer.mfgcs is not None, "optimizer.mfgcs is required for sampler='mfgcs'"
        self._cfg = cfg
        self._dataset_builder = dataset_builder
        self._mfgcs_cfg = cfg.optimizer.mfgcs
        self._search_space: Dict[str, SearchSpaceParam] = cfg.optimizer.search_space
        self._scene_sampler = scene_sampler_factory(self._mfgcs_cfg.scene_sampler)
        self._coord_optimizer = coordinate_optimizer_factory(self._mfgcs_cfg.coordinate_optimizer)
        self._shrinker = SearchSpaceShrinker(self._mfgcs_cfg.shrink)
        self._trial_counter = 0
        self._all_trials: List[TrialResult] = []
        self._best_trial: Optional[TrialResult] = None

    # --- Public entry point -------------------------------------------------

    def run(self) -> None:
        cfg = self._cfg
        guard_optimization_dir(cfg, cfg.optimizer.study_name)
        cfg.inference.override = True
        bootstrap_detection_cache(cfg, dataset_builder=self._dataset_builder)

        all_scenes = self._dataset_builder(cfg).scenes

        current = extract_base_params(cfg, self._search_space)
        windows: Dict[str, SearchWindow] = {
            dp: SearchWindow.from_spec(spec) for dp, spec in self._search_space.items()
        }

        if self._mfgcs_cfg.bootstrap_full_eval:
            best_score = self._record_full_eval(current, params=current, label='bootstrap')
        else:
            best_score = float('-inf')

        history: List[MFGCSSweepRecord] = []
        for sweep_idx in range(self._mfgcs_cfg.max_sweeps):
            sweep_records: List[MFGCSCoordinateRecord] = []
            improved = False

            for coord_idx, (dotpath, spec) in enumerate(self._search_space.items()):
                window = self._effective_window(dotpath, spec, current, windows)
                subset = self._scene_sampler.sample(all_scenes)
                candidate, low_score = self._optimize_coordinate(
                    dotpath, spec, current, window, subset
                )
                record = MFGCSCoordinateRecord(
                    sweep=sweep_idx,
                    coord_index=coord_idx,
                    dotpath=dotpath,
                    previous_value=current[dotpath],
                    candidate_value=candidate,
                    accepted=False,
                    low_score=low_score,
                    full_score=None,
                    sampled_scenes=list(subset),
                )
                if _values_equal(candidate, current[dotpath]):
                    record.skipped_full_eval = True
                    record.note = 'no-change'
                    sweep_records.append(record)
                    continue

                full_params = {**current, dotpath: candidate}
                full_score = self._record_full_eval(
                    full_params,
                    params={dotpath: candidate},
                    label=f'sweep{sweep_idx}-coord{coord_idx}-{dotpath}',
                )
                record.full_score = full_score

                if full_score > best_score:
                    current[dotpath] = candidate
                    best_score = full_score
                    record.accepted = True
                    improved = True
                    windows[dotpath] = self._shrinker.shrink(spec, candidate, windows[dotpath])
                    logger.info(
                        f'[sweep {sweep_idx} / coord {coord_idx}] ACCEPT {dotpath}: '
                        f'{record.previous_value} -> {candidate} (full HOTA={full_score:.4f})'
                    )
                else:
                    logger.info(
                        f'[sweep {sweep_idx} / coord {coord_idx}] reject {dotpath}: '
                        f'{record.previous_value} -> {candidate} '
                        f'(full HOTA={full_score:.4f} <= best {best_score:.4f})'
                    )
                sweep_records.append(record)

            history.append(MFGCSSweepRecord(
                sweep=sweep_idx,
                accepted_count=sum(1 for r in sweep_records if r.accepted),
                coordinates=sweep_records,
            ))

            if not improved:
                logger.info(f'No improvement in sweep {sweep_idx}; stopping.')
                break

        self._save_results(history)

    # --- Internals ----------------------------------------------------------

    def _effective_window(
        self,
        dotpath: str,
        spec: SearchSpaceParam,
        current: Dict[str, Any],
        windows: Dict[str, SearchWindow],
    ) -> SearchWindow:
        """Apply ``min_param`` / ``max_param`` constraints from current values."""
        base = windows[dotpath]
        if spec.type == 'categorical':
            return base

        low = base.low
        high = base.high
        if spec.min_param is not None and spec.min_param in current:
            dep_low = float(current[spec.min_param])
            low = dep_low if low is None else max(float(low), dep_low)
        if spec.max_param is not None and spec.max_param in current:
            dep_high = float(current[spec.max_param])
            high = dep_high if high is None else min(float(high), dep_high)
        return SearchWindow(low=low, high=high)

    def _optimize_coordinate(
        self,
        dotpath: str,
        spec: SearchSpaceParam,
        current: Dict[str, Any],
        window: SearchWindow,
        subset: List[str],
    ) -> tuple:
        """Run the coordinate optimizer; return (candidate_value, last_low_score)."""
        last_score: Dict[str, Optional[float]] = {'value': None}

        def low_eval(value: Any) -> float:
            params = {**current, dotpath: value}
            score = evaluate(
                self._cfg,
                params,
                scenes=subset,
                dataset_builder=self._dataset_builder,
            )
            last_score['value'] = score
            return score

        candidate = self._coord_optimizer.optimize(
            spec, current[dotpath], low_eval, window=window
        )
        return candidate, last_score['value']

    def _record_full_eval(
        self,
        full_params: Dict[str, Any],
        params: Dict[str, Any],
        label: str,
    ) -> float:
        """Evaluate ``full_params`` on the full split, log to MLflow, store TrialResult."""
        cfg = self._cfg
        trial_cfg = cfg.override(full_params)
        trial_cfg.inference.override = True

        optuna_data = OptunaOutputData(
            study_name=cfg.optimizer.study_name,
            trial_number=self._trial_counter,
            trial_params=params,
        )
        score = evaluate(
            cfg,
            full_params,
            scenes=None,
            optuna_data=optuna_data,
            dataset_builder=self._dataset_builder,
        )
        log_trial_to_mlflow(trial_cfg, optuna_data)

        trial = TrialResult(
            number=self._trial_counter,
            value=score,
            params=dict(full_params),
            state='COMPLETE',
            config_hash=trial_cfg.hash,
        )
        self._all_trials.append(trial)
        if self._best_trial is None or score > self._best_trial.value:
            self._best_trial = trial
        logger.info(f'Full-fidelity eval [{label}]: HOTA={score:.4f}')
        self._trial_counter += 1
        return score

    def _save_results(self, history: List[MFGCSSweepRecord]) -> None:
        cfg = self._cfg
        assert self._best_trial is not None, 'No trials were evaluated'
        results = OptimizationResults(
            study_name=cfg.optimizer.study_name,
            algorithm='mfgcs',
            best_trial=self._best_trial,
            all_trials=list(self._all_trials),
            extras={
                'mfgcs_history': [s.to_dict() for s in history],
                'mfgcs_config': {
                    'scene_sampler': {
                        'type': self._mfgcs_cfg.scene_sampler.type,
                        'params': dict(self._mfgcs_cfg.scene_sampler.params),
                    },
                    'coordinate_optimizer': {
                        'type': self._mfgcs_cfg.coordinate_optimizer.type,
                        'params': dict(self._mfgcs_cfg.coordinate_optimizer.params),
                    },
                    'max_sweeps': self._mfgcs_cfg.max_sweeps,
                    'bootstrap_full_eval': self._mfgcs_cfg.bootstrap_full_eval,
                    'shrink': {
                        'enabled': self._mfgcs_cfg.shrink.enabled,
                        'radius_frac': self._mfgcs_cfg.shrink.radius_frac,
                        'window_size': self._mfgcs_cfg.shrink.window_size,
                    },
                },
            },
        )
        split_path = conventions.get_split_results_path(
            master_path=cfg.path.master,
            dataset_type=cfg.dataset.type,
            experiment_name=cfg.experiment,
            split=cfg.inference.split,
            dataset_name=cfg.dataset.name,
        )
        results_path = conventions.get_optimization_results_path(split_path, cfg.optimizer.study_name)
        results.save(results_path)
        logger.info(
            f'Best trial #{self._best_trial.number}: HOTA={self._best_trial.value:.4f}, '
            f'params={self._best_trial.params}'
        )
        logger.info(f'Optimization results saved to "{results_path}".')
