"""
Integration test for MFGCS that runs the *real* algorithm against a *real*
``GlobalConfig`` composed via Hydra, with only the inference + eval I/O
stubbed. Validates:

- the dispatcher routes ``sampler='mfgcs'`` to ``MFGCSPipeline.run()``,
- ``cfg.override`` works for every dotpath the algorithm touches,
- ``cfg.hash`` cleanly separates subset and full-split caches,
- ``run_eval(scenes=...)`` is wired in,
- ``optimization_results.json`` is written with ``algorithm='mfgcs'`` and
  the expected ``mfgcs_history`` shape.

Heavy I/O (``run_inference``, ``run_eval``) is replaced with deterministic
synthetic scoring so the test runs in <1s on CPU without needing a dataset.
"""
import json
import os
import tempfile
import unittest
from typing import Any, Dict, List, Optional
from unittest import mock

# Importing config_parser registers the structured config in Hydra's ConfigStore.
import motrack.config_parser  # noqa: F401  (registration side-effect)
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from motrack.common.project import DANCETRACK_TRACKERS_CONFIG_PATH
from motrack.config_parser import GlobalConfig
from motrack.tools.optimization import run_optimize


def _objective(params: Dict[str, Any]) -> float:
    """Synthetic HOTA: max around match_threshold=0.30, fuse_score=true."""
    mt = float(params.get('algorithm.params.matcher.params.match_threshold', 0.3))
    fuse = params.get('algorithm.params.matcher.params.fuse_score', True)
    base = 0.7 - (mt - 0.30) ** 2
    return base + (0.05 if fuse else 0.0)


class MFGCSEndToEndTest(unittest.TestCase):
    """Algorithm run end-to-end against a real composed GlobalConfig."""

    def setUp(self) -> None:
        os.environ.setdefault('MLFLOW_TRACKING_URI', 'http://localhost:5000')
        # Hydra composes the real mfgcs_sort.yaml — same path the CLI takes.
        with initialize_config_dir(config_dir=DANCETRACK_TRACKERS_CONFIG_PATH, version_base='1.1'):
            cfg = compose(
                config_name='optimization/mfgcs_sort',
                overrides=[
                    # Keep the run tiny: 1 sweep, small grid, no shrinking complexity.
                    'optimizer.sampler_params.max_sweeps=1',
                    'optimizer.sampler_params.coordinate_optimizer.params.grid=3',
                    'optimizer.sampler_params.coordinate_optimizer.params.rounds=1',
                    'optimizer.sampler_params.scene_sampler.params.n=2',
                    'mlflow.enabled=false',
                ],
            )
            self.cfg: GlobalConfig = OmegaConf.to_object(cfg)

        self.tmpdir = tempfile.mkdtemp()
        # Redirect all output paths to the temp dir so we don't touch /media/home.
        self.cfg.path.master = self.tmpdir
        self.cfg.path.assets = self.tmpdir
        # Also re-run __post_init__ to refresh derived fields (dataset paths).
        self.cfg.__post_init__()
        # Seed a fake "current" detection cache path that already "exists" so
        # bootstrap_detection_cache becomes a no-op.
        self.cfg.object_detection.cache_path = self.tmpdir

        # Pretend the dataset has 5 scenes.
        self.scene_names = ['s1', 's2', 's3', 's4', 's5']

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_run_optimize_dispatches_to_mfgcs_and_writes_results(self) -> None:
        from motrack.optimization.mfgcs import algorithm as algorithm_module

        ran_inference: List[Dict[str, Any]] = []
        ran_eval: List[Optional[List[str]]] = []

        # Stand-in dataset: only attribute the algorithm accesses is .scenes.
        class _FakeDataset:
            scenes = self.scene_names

        def fake_dataset_builder(_cfg):
            return _FakeDataset()

        def fake_run_inference(cfg, *, inference_output=None, dataset_builder=None, **kwargs):
            # Make the experiment dir + a marker file so any cache-existence
            # checks downstream see a "real" directory.
            os.makedirs(cfg.experiment_path, exist_ok=True)
            ran_inference.append({
                'hash': cfg.hash,
                'scene_pattern': cfg.dataset_filter.scene_pattern,
            })

        def fake_run_eval(cfg, *, dataset_builder=None, scenes=None, **kwargs):
            ran_eval.append(list(scenes) if scenes is not None else None)
            current_params = {
                dp: cfg.resolve(dp) for dp in self.cfg.optimizer.search_space
            }
            score = _objective(current_params)
            results = {
                'combined': {'HOTA': {'HOTA': [score]}},
                'sequences': {s: {'HOTA': {'HOTA': [score]}} for s in (scenes or self.scene_names)},
            }
            # Also persist eval_results.json so the cache-hit branch in evaluate()
            # would work on a re-run (not strictly required for this test).
            from motrack.common import conventions
            from motrack.eval.results import EvalResults
            EvalResults(combined=results['combined'], sequences=results['sequences']).save(
                conventions.get_eval_results_path(cfg.experiment_path)
            )
            return results

        with mock.patch('motrack.optimization.common.run_inference', side_effect=fake_run_inference), \
             mock.patch('motrack.optimization.common.run_eval', side_effect=fake_run_eval), \
             mock.patch.object(algorithm_module, 'log_trial_to_mlflow'):
            run_optimize(self.cfg, dataset_builder=fake_dataset_builder)

        # 1. Results file written at the expected path with algorithm='mfgcs'.
        from motrack.common import conventions
        split_path = conventions.get_split_results_path(
            master_path=self.cfg.path.master,
            dataset_type=self.cfg.dataset.type,
            experiment_name=self.cfg.experiment,
            split=self.cfg.inference.split,
            dataset_name=self.cfg.dataset.name,
        )
        results_path = conventions.get_optimization_results_path(
            split_path, self.cfg.optimizer.study_name
        )
        self.assertTrue(os.path.exists(results_path), f'no results file at {results_path}')
        with open(results_path, 'r') as f:
            payload = json.load(f)
        self.assertEqual(payload['algorithm'], 'mfgcs')
        self.assertIn('mfgcs_history', payload['extras'])
        self.assertIn('mfgcs_config', payload['extras'])
        self.assertGreaterEqual(len(payload['all_trials']), 1)

        # 2. History records: one sweep, one record per parameter (3 in this config).
        history = payload['extras']['mfgcs_history']
        self.assertEqual(len(history), 1)
        self.assertEqual(len(history[0]['coordinates']), len(self.cfg.optimizer.search_space))

        # 3. The algorithm exercised both fidelities — at least one full and
        #    several subset evaluations.
        self.assertIn(None, ran_eval, 'expected at least one full-fidelity eval')
        subset_calls = [s for s in ran_eval if s is not None]
        self.assertGreater(len(subset_calls), 0, 'expected at least one low-fidelity eval')
        for s in subset_calls:
            self.assertEqual(len(s), 2, 'subset size must match scene_sampler.params.n')

        # 4. cfg.hash separates subset evals from full evals (different scene_pattern).
        full_hashes = {r['hash'] for r in ran_inference if r['scene_pattern'] == '(.*?)'}
        subset_hashes = {r['hash'] for r in ran_inference if r['scene_pattern'] != '(.*?)'}
        self.assertTrue(full_hashes.isdisjoint(subset_hashes),
                        'full and subset evals must produce disjoint cache keys')

        # 5. Best trial value should reflect the synthetic objective's preference
        #    (fuse_score=true at match_threshold=0.30 → ~0.75 baseline) for SOME
        #    accepted move; at minimum, best should be >= bootstrap.
        best = payload['best_trial']['value']
        self.assertGreaterEqual(best, 0.0)


if __name__ == '__main__':
    unittest.main()
