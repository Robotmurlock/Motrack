"""
Smoke / contract tests for ``MFGCSAlgorithm``.

The full algorithm runs inference + eval, so these tests stub the
inference/eval boundary (``evaluate``) and supporting I/O. Goals:

- accept-gate: only strict full-fidelity improvements update ``current``;
- stopping rule: a sweep with no accepted move terminates the algorithm;
- dependent-param bounds: ``min_param`` / ``max_param`` shrink the window
  used for the coordinate optimizer at evaluation time;
- history shape: per-coordinate records and per-sweep aggregates are recorded.
"""
import types
import unittest
from typing import Any, Dict, List, Optional, Tuple
from unittest import mock

from motrack.config_parser import (
    FactorySpec,
    MFGCSConfig,
    MFGCSShrinkConfig,
    SearchSpaceParam,
)
from motrack.tools.optimization.mfgcs import algorithm as algorithm_module
from motrack.tools.optimization.mfgcs.algorithm import MFGCSAlgorithm
from motrack.tools.optimization.mfgcs.coordinate import SearchWindow


def _stub_cfg(search_space: Dict[str, SearchSpaceParam], mfgcs_cfg: MFGCSConfig, base: Dict[str, Any]):
    """Build a minimal cfg-like SimpleNamespace for the algorithm.

    The algorithm only needs:
    - ``cfg.optimizer.mfgcs`` (typed)
    - ``cfg.optimizer.search_space`` (dict)
    - ``cfg.optimizer.study_name``
    - ``cfg.inference.override`` (writable)
    - ``cfg.resolve(dotpath)`` for base-param extraction
    - ``cfg.override(dict)`` returning a stand-in trial_cfg with ``.hash`` and writable ``inference.override``
    """
    optimizer = types.SimpleNamespace(
        mfgcs=mfgcs_cfg,
        search_space=search_space,
        study_name='test_study',
        sampler='mfgcs',
    )

    def resolve(dotpath: str) -> Any:
        return base[dotpath]

    def override(d: Dict[str, Any]):
        merged = {**base, **d}
        return types.SimpleNamespace(
            hash='hash-' + '-'.join(f'{k}={v}' for k, v in sorted(merged.items())),
            inference=types.SimpleNamespace(override=False),
            optimizer=optimizer,
        )

    return types.SimpleNamespace(
        optimizer=optimizer,
        inference=types.SimpleNamespace(override=False),
        resolve=resolve,
        override=override,
    )


def _stub_dataset_builder(scenes: List[str]):
    def builder(_cfg):
        return types.SimpleNamespace(scenes=list(scenes))
    return builder


class MFGCSAlgorithmTest(unittest.TestCase):
    """Algorithm contract tests with mocked I/O boundary."""

    def setUp(self) -> None:
        self.search_space = {
            'a': SearchSpaceParam(type='float', low=0.0, high=1.0),
            'b': SearchSpaceParam(type='int', low=0, high=10),
        }
        self.base = {'a': 0.1, 'b': 1}
        # Coarse-to-fine + random sampler → deterministic enough for an end-to-end check.
        self.mfgcs_cfg = MFGCSConfig(
            scene_sampler=FactorySpec(type='random', params={'n': 2, 'seed': 0}),
            coordinate_optimizer=FactorySpec(type='coarse_to_fine', params={'grid': 5, 'rounds': 2}),
            max_sweeps=3,
            bootstrap_full_eval=True,
            shrink=MFGCSShrinkConfig(enabled=False),  # turn off shrinking to keep windows simple
        )
        self.cfg = _stub_cfg(self.search_space, self.mfgcs_cfg, self.base)
        self.dataset_builder = _stub_dataset_builder(['s1', 's2', 's3', 's4', 's5'])

    def _run_with_scripted_scores(
        self,
        evaluate_side_effect,
        max_sweeps: Optional[int] = None,
    ) -> MFGCSAlgorithm:
        """Run the algorithm with a scripted ``evaluate`` and capture state.

        Returns the constructed algorithm so the caller can inspect trials.
        """
        if max_sweeps is not None:
            self.mfgcs_cfg.max_sweeps = max_sweeps

        with mock.patch.object(algorithm_module, 'evaluate', side_effect=evaluate_side_effect), \
             mock.patch.object(algorithm_module, 'bootstrap_detection_cache'), \
             mock.patch.object(algorithm_module, 'guard_optimization_dir', return_value='/tmp/test_split'), \
             mock.patch.object(algorithm_module, 'log_trial_to_mlflow'), \
             mock.patch.object(MFGCSAlgorithm, '_save_results') as save_mock:
            algo = MFGCSAlgorithm(self.cfg, dataset_builder=self.dataset_builder)
            algo.run()
            self.history = save_mock.call_args[0][0]  # list[MFGCSSweepRecord]
        return algo

    def test_accept_only_on_strict_improvement(self) -> None:
        # Low-fidelity always returns 0; full-fidelity returns:
        #   - 0.5 for the bootstrap (current state),
        #   - 0.6 when 'a' is being moved (accepted),
        #   - 0.4 when 'b' is being moved (rejected).
        full_calls: List[Dict[str, Any]] = []

        def side_effect(_cfg, overrides, scenes=None, **kwargs):
            if scenes is not None:
                return 0.0
            full_calls.append(dict(overrides))
            # Bootstrap is the first call where overrides equal the base values.
            if overrides == {'a': 0.1, 'b': 1}:
                return 0.5
            return 0.6 if 'a' in overrides and overrides['a'] != 0.1 else 0.4

        algo = self._run_with_scripted_scores(side_effect)
        self.assertEqual(algo._best_trial.value, 0.6)
        self.assertGreaterEqual(len(algo._all_trials), 2)
        self.assertEqual(self.history[0].accepted_count, 1)

        accepted = [r for r in self.history[0].coordinates if r.accepted]
        rejected = [r for r in self.history[0].coordinates if not r.accepted and not r.skipped_full_eval]
        self.assertEqual([r.dotpath for r in accepted], ['a'])
        self.assertIn('b', [r.dotpath for r in rejected])

    def test_no_improvement_stops_after_one_sweep(self) -> None:
        def side_effect(_cfg, overrides, scenes=None, **kwargs):
            if scenes is not None:
                return 0.0
            # Bootstrap returns 0.9; everything else is worse.
            return 0.9 if overrides == {'a': 0.1, 'b': 1} else 0.0

        algo = self._run_with_scripted_scores(side_effect, max_sweeps=5)
        self.assertEqual(len(self.history), 1, 'should stop after the first no-improvement sweep')
        self.assertEqual(self.history[0].accepted_count, 0)
        self.assertEqual(algo._best_trial.value, 0.9)

    def test_history_shape(self) -> None:
        def side_effect(_cfg, overrides, scenes=None, **kwargs):
            return 0.0

        self._run_with_scripted_scores(side_effect, max_sweeps=2)
        sweep0 = self.history[0]
        self.assertEqual(len(sweep0.coordinates), len(self.search_space))
        for rec in sweep0.coordinates:
            self.assertIn(rec.dotpath, self.search_space)
            self.assertEqual(len(rec.sampled_scenes), 2)


class MFGCSDependentParamTest(unittest.TestCase):
    """``min_param`` / ``max_param`` constraints applied at coordinate time."""

    def test_min_param_constrains_coordinate_window(self) -> None:
        search_space = {
            'a': SearchSpaceParam(type='float', low=0.0, high=1.0),
            'b': SearchSpaceParam(type='float', low=0.0, high=1.0, min_param='a'),
        }
        base = {'a': 0.7, 'b': 0.8}
        mfgcs_cfg = MFGCSConfig(
            scene_sampler=FactorySpec(type='random', params={'n': 2, 'seed': 0}),
            coordinate_optimizer=FactorySpec(type='coarse_to_fine', params={'grid': 5, 'rounds': 1}),
            max_sweeps=1,
            # Bootstrap with a high score so subsequent 0-scoring full evals all reject —
            # 'a' stays at 0.7, so the min_param constraint for 'b' is well-defined.
            bootstrap_full_eval=True,
            shrink=MFGCSShrinkConfig(enabled=False),
        )
        cfg = _stub_cfg(search_space, mfgcs_cfg, base)
        evaluated_b_inputs: List[float] = []

        def side_effect(_cfg, overrides, scenes=None, **kwargs):
            # Track only the values 'b' takes during low-fidelity coordinate search.
            if scenes is not None and overrides.get('a') == 0.7:
                evaluated_b_inputs.append(overrides['b'])
            # Bootstrap (full eval, both at base) returns 1.0; everything else 0.0.
            if scenes is None and overrides == {'a': 0.7, 'b': 0.8}:
                return 1.0
            return 0.0

        with mock.patch.object(algorithm_module, 'evaluate', side_effect=side_effect), \
             mock.patch.object(algorithm_module, 'bootstrap_detection_cache'), \
             mock.patch.object(algorithm_module, 'guard_optimization_dir', return_value='/tmp/x'), \
             mock.patch.object(algorithm_module, 'log_trial_to_mlflow'), \
             mock.patch.object(MFGCSAlgorithm, '_save_results'):
            algo = MFGCSAlgorithm(cfg, dataset_builder=_stub_dataset_builder(['s1', 's2', 's3']))
            algo.run()

        # All evaluated values for 'b' must respect min_param=a (currently 0.7)
        self.assertGreater(len(evaluated_b_inputs), 0)
        for v in evaluated_b_inputs:
            self.assertGreaterEqual(v, 0.7 - 1e-9, f'b={v} violated min_param=a=0.7')


if __name__ == '__main__':
    unittest.main()
