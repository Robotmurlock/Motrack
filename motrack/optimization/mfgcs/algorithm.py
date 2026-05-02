"""
Multi-Fidelity Greedy Coordinate Search (MFGCS) algorithm.

Outer loop sweeps parameters, optimizes each via a low-fidelity scene subset,
validates the candidate on the full split, and accepts only strict
improvements. The sweep ends once a full pass produces no accepted move.
"""
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

from motrack.common import conventions
from motrack.config_parser import GlobalConfig, SearchSpaceParam
from motrack.tools.dataset_builder import DatasetBuilder
from motrack.tools.inference import OptunaOutputData
from motrack.optimization.base import OptimizationPipeline
from motrack.optimization.common import (
    bootstrap_detection_cache,
    evaluate,
    extract_base_params,
    guard_optimization_dir,
    is_eval_cached,
    log_trial_to_mlflow,
)
from motrack.optimization.mfgcs.coordinate import (
    SearchWindow,
    coordinate_optimizer_factory,
)
from motrack.optimization.mfgcs.params import MFGCSParams
from motrack.optimization.mfgcs.results import (
    MFGCSCoordinateRecord,
    MFGCSSweepRecord,
)
from motrack.optimization.mfgcs.scene_sampler import scene_sampler_factory
from motrack.optimization.mfgcs.shrinking import SearchSpaceShrinker
from motrack.optimization.results import OptimizationResults, TrialResult

logger = logging.getLogger('Tool-Optimize-MFGCS')


def _values_equal(a: Any, b: Any) -> bool:
    """Equality with FP tolerance for floats; strict equality otherwise."""
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
        except (TypeError, ValueError):
            return a == b
    return a == b


class MFGCSPipeline(OptimizationPipeline):
    """Driver for one MFGCS optimization run."""

    def __init__(
        self,
        cfg: GlobalConfig,
        dataset_builder: DatasetBuilder,
        params: MFGCSParams,
    ) -> None:
        super().__init__(cfg, dataset_builder, params)
        self._mfgcs_cfg: MFGCSParams = params
        self._search_space: Dict[str, SearchSpaceParam] = cfg.optimizer.search_space
        self._scene_sampler = scene_sampler_factory(self._mfgcs_cfg.scene_sampler)
        self._coord_optimizer = coordinate_optimizer_factory(self._mfgcs_cfg.coordinate_optimizer)
        self._shrinker = SearchSpaceShrinker(self._mfgcs_cfg.shrink)
        self._trial_counter = 0
        self._all_trials: List[TrialResult] = []
        self._best_trial: Optional[TrialResult] = None
        # Coord-search work that hasn't been attributed to a full-fidelity
        # trial yet (because the coord optimizer returned the current value
        # and the algorithm skipped the acceptance eval). Folded into the
        # next TrialResult that does occur.
        self._pending_scenes = 0
        self._pending_time_s = 0.0
        self._n_full_scenes: Optional[int] = None

    # --- Public entry point -------------------------------------------------

    def run(self) -> None:
        cfg = self._cfg
        guard_optimization_dir(cfg, cfg.optimizer.study_name)
        cfg.inference.override = True
        bootstrap_detection_cache(cfg, dataset_builder=self._dataset_builder)

        all_scenes = self._dataset_builder(cfg).scenes
        self._n_full_scenes = len(all_scenes)

        current = extract_base_params(cfg, self._search_space)
        windows: Dict[str, SearchWindow] = {
            dp: SearchWindow.from_spec(spec) for dp, spec in self._search_space.items()
        }
        # Per-parameter consecutive barren-sweep counter. After
        # ``drop_after_barren_sweeps`` consecutive sweeps with no accept on a
        # parameter, drop it from the active set to save budget.
        barren_streak: Dict[str, int] = {dp: 0 for dp in self._search_space}
        dropped: set = set()
        accept_threshold = float(self._mfgcs_cfg.accept_threshold)
        drop_after = int(self._mfgcs_cfg.drop_after_barren_sweeps)
        max_trials = int(self._mfgcs_cfg.max_trials)

        if self._mfgcs_cfg.bootstrap_full_eval:
            best_score = self._record_full_eval(current, params=current, label='bootstrap')
        else:
            best_score = float('-inf')

        def budget_exhausted() -> bool:
            return max_trials > 0 and self._trial_counter >= max_trials

        history: List[MFGCSSweepRecord] = []
        for sweep_idx in range(self._mfgcs_cfg.max_sweeps):
            if budget_exhausted():
                logger.info(
                    f'Reached max_trials={max_trials} (full-fidelity evals); '
                    f'stopping at start of sweep {sweep_idx}.'
                )
                break

            sweep_records: List[MFGCSCoordinateRecord] = []
            improved = False
            sweep_accepted_params: set = set()

            active_params = [
                (i, dp, sp) for i, (dp, sp) in enumerate(self._search_space.items())
                if dp not in dropped
            ]
            if not active_params:
                logger.info(
                    f'All parameters dropped after barren-sweep limit '
                    f'({drop_after}); stopping at sweep {sweep_idx}.'
                )
                break

            for coord_idx, dotpath, spec in active_params:
                if budget_exhausted():
                    logger.info(
                        f'Reached max_trials={max_trials}; '
                        f'stopping mid-sweep {sweep_idx} at coord {coord_idx}.'
                    )
                    break
                window = self._effective_window(dotpath, spec, current, windows)
                subset = self._scene_sampler.sample(all_scenes)
                candidate, low_score, coord_scenes, coord_time = self._optimize_coordinate(
                    dotpath, spec, current, window, subset
                )
                self._pending_scenes += coord_scenes
                self._pending_time_s += coord_time
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

                gain = full_score - best_score
                if gain > accept_threshold:
                    current[dotpath] = candidate
                    best_score = full_score
                    record.accepted = True
                    improved = True
                    sweep_accepted_params.add(dotpath)
                    windows[dotpath] = self._shrinker.shrink(spec, candidate, windows[dotpath])
                    logger.info(
                        f'[sweep {sweep_idx} / coord {coord_idx}] ACCEPT {dotpath}: '
                        f'{record.previous_value} -> {candidate} '
                        f'(full HOTA={full_score:.4f}, gain={gain:+.4f} > thr={accept_threshold:.4f})'
                    )
                else:
                    logger.info(
                        f'[sweep {sweep_idx} / coord {coord_idx}] reject {dotpath}: '
                        f'{record.previous_value} -> {candidate} '
                        f'(full HOTA={full_score:.4f}, gain={gain:+.4f} <= thr={accept_threshold:.4f})'
                    )
                sweep_records.append(record)

            # Update barren streaks for every active parameter.
            if drop_after > 0:
                for _, dotpath, _ in active_params:
                    if dotpath in sweep_accepted_params:
                        barren_streak[dotpath] = 0
                    else:
                        barren_streak[dotpath] += 1
                        if barren_streak[dotpath] >= drop_after:
                            dropped.add(dotpath)
                            logger.info(
                                f'Dropping {dotpath} after {drop_after} consecutive '
                                f'barren sweeps; remaining active: {len(self._search_space) - len(dropped)}'
                            )

            history.append(MFGCSSweepRecord(
                sweep=sweep_idx,
                accepted_count=sum(1 for r in sweep_records if r.accepted),
                coordinates=sweep_records,
            ))

            if not improved:
                if self._mfgcs_cfg.early_stop:
                    logger.info(f'No improvement in sweep {sweep_idx}; stopping (early_stop=True).')
                    break
                logger.info(
                    f'No improvement in sweep {sweep_idx}; continuing (early_stop=False). '
                    f'Subsequent sweeps draw fresh scene subsets.'
                )

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
    ) -> Tuple[Any, Optional[float], int, float]:
        """Run the coordinate optimizer.

        Returns ``(candidate_value, last_low_score, scenes_evaluated, wall_time_s)``
        where the latter two count the low-fidelity work performed across all
        ``low_eval`` invocations the coordinate optimizer made.
        """
        last_score: Dict[str, Optional[float]] = {'value': None}
        scenes_used = 0
        time_used = 0.0
        subset_size = len(subset)

        def low_eval(value: Any) -> float:
            nonlocal scenes_used, time_used
            params = {**current, dotpath: value}
            cache_hit = is_eval_cached(self._cfg, params, scenes=subset)
            t0 = time.perf_counter()
            score = evaluate(
                self._cfg,
                params,
                scenes=subset,
                dataset_builder=self._dataset_builder,
            )
            time_used += time.perf_counter() - t0
            if not cache_hit:
                scenes_used += subset_size
            last_score['value'] = score
            return score

        candidate = self._coord_optimizer.optimize(
            spec, current[dotpath], low_eval, window=window
        )
        return candidate, last_score['value'], scenes_used, time_used

    def _record_full_eval(
        self,
        full_params: Dict[str, Any],
        params: Dict[str, Any],
        label: str,
    ) -> float:
        """Evaluate ``full_params`` on the full split, log to MLflow, store TrialResult.

        The trial's ``scenes_evaluated`` and ``wall_time_s`` include any
        coord-search work that has been buffered in ``_pending_*`` since the
        previous full-fidelity eval, so cumulative budgets remain accurate
        even when intermediate coord steps produced no candidate change.
        """
        cfg = self._cfg
        trial_cfg = cfg.override(full_params)
        trial_cfg.inference.override = True

        optuna_data = OptunaOutputData(
            study_name=cfg.optimizer.study_name,
            trial_number=self._trial_counter,
            trial_params=params,
        )
        full_scenes = self._n_full_scenes if self._n_full_scenes is not None else 0
        cache_hit = is_eval_cached(cfg, full_params, scenes=None)
        t0 = time.perf_counter()
        score = evaluate(
            cfg,
            full_params,
            scenes=None,
            optuna_data=optuna_data,
            dataset_builder=self._dataset_builder,
        )
        full_time = time.perf_counter() - t0

        scenes_evaluated = self._pending_scenes + (0 if cache_hit else full_scenes)
        wall_time_s = self._pending_time_s + full_time
        self._pending_scenes = 0
        self._pending_time_s = 0.0

        log_trial_to_mlflow(
            trial_cfg,
            optuna_data,
            extra_metrics={
                'scenes_evaluated': float(scenes_evaluated),
                'trial_wall_time_s': float(wall_time_s),
            },
        )

        trial = TrialResult(
            number=self._trial_counter,
            value=score,
            params=dict(full_params),
            state='COMPLETE',
            config_hash=trial_cfg.hash,
            scenes_evaluated=scenes_evaluated,
            wall_time_s=wall_time_s,
        )
        self._all_trials.append(trial)
        if self._best_trial is None or score > self._best_trial.value:
            self._best_trial = trial
        logger.info(
            f'Full-fidelity eval [{label}]: HOTA={score:.4f}, '
            f'scenes={scenes_evaluated}, wall_time={wall_time_s:.1f}s'
        )
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
                    'max_trials': self._mfgcs_cfg.max_trials,
                    'bootstrap_full_eval': self._mfgcs_cfg.bootstrap_full_eval,
                    'early_stop': self._mfgcs_cfg.early_stop,
                    'accept_threshold': self._mfgcs_cfg.accept_threshold,
                    'drop_after_barren_sweeps': self._mfgcs_cfg.drop_after_barren_sweeps,
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
