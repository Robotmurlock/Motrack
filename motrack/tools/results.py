"""
Aggregators that load all tool outputs for an experiment.

``TrackerRunResult`` collects the artifacts of a single config-hash run
(``run_meta.json``, ``eval_results.json``, lazy-loaded ``fps_stats`` and
config snapshot). ``ExperimentResults`` walks a split directory and loads
every run plus all optimization studies under it.
"""
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

from motrack.common import conventions
from motrack.eval.results import EvalResults
from motrack.tools.inference import InferenceOutputData
from motrack.tools.optimization import OptimizationResults


@dataclass
class TrackerRunResult:
    """Complete result for a single tracker run (one config-hash directory)."""
    config_hash: str
    run_dir: str
    inference_output: InferenceOutputData
    eval_results: Optional[EvalResults] = None
    _fps_stats: Optional[Dict[str, Any]] = field(default=None, repr=False)
    _config_snapshot: Optional[Dict[str, Any]] = field(default=None, repr=False)
    _fps_loaded: bool = field(default=False, repr=False)
    _config_loaded: bool = field(default=False, repr=False)

    @property
    def fps_stats(self) -> Optional[Dict[str, Any]]:
        """Lazily load fps_stats.json on first access."""
        if not self._fps_loaded:
            fps_path = conventions.get_fps_stats_path(self.run_dir)
            if os.path.exists(fps_path):
                with open(fps_path, 'r', encoding='utf-8') as f:
                    self._fps_stats = json.load(f)
            self._fps_loaded = True
        return self._fps_stats

    @property
    def config_snapshot(self) -> Optional[Dict[str, Any]]:
        """Lazily load config.yaml on first access."""
        if not self._config_loaded:
            config_path = conventions.get_config_snapshot_path(self.run_dir)
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    self._config_snapshot = yaml.safe_load(f)
            self._config_loaded = True
        return self._config_snapshot

    @classmethod
    def load(cls, run_dir: str) -> 'TrackerRunResult':
        """Load core artifacts from a config-hash directory.

        FPS stats and config snapshots are loaded lazily on first access.
        """
        config_hash = os.path.basename(run_dir)

        inference_output = InferenceOutputData.load(conventions.get_run_meta_path(run_dir))

        eval_path = conventions.get_eval_results_path(run_dir)
        eval_results = EvalResults.load(eval_path) if os.path.exists(eval_path) else None

        return cls(
            config_hash=config_hash,
            run_dir=run_dir,
            inference_output=inference_output,
            eval_results=eval_results,
        )


@dataclass
class ExperimentResults:
    """All runs under an experiment/split, with optimization results."""
    experiment_name: str
    dataset_name: str
    split: str
    runs: List[TrackerRunResult] = field(default_factory=list)
    optimizations: Dict[str, OptimizationResults] = field(default_factory=dict)

    @classmethod
    def collect(cls, split_path: str) -> 'ExperimentResults':
        """
        Walk a split directory and load all tracker run results.

        Args:
            split_path: ``{master}/{dataset}/{experiment}/{split}/``

        Returns:
            Populated ``ExperimentResults`` with all runs and optimization
            results.
        """
        split = os.path.basename(split_path)
        experiment_name = os.path.basename(os.path.dirname(split_path))
        dataset_name = os.path.basename(os.path.dirname(os.path.dirname(split_path)))

        runs: List[TrackerRunResult] = []
        inference_path = conventions.get_inference_path(split_path)
        if os.path.isdir(inference_path):
            for entry in sorted(os.listdir(inference_path)):
                entry_path = os.path.join(inference_path, entry)
                if not os.path.isdir(entry_path):
                    continue
                inference_output_path = conventions.get_run_meta_path(entry_path)
                if not os.path.exists(inference_output_path):
                    continue
                runs.append(TrackerRunResult.load(entry_path))

        optimizations: Dict[str, OptimizationResults] = {}
        optimizations_dir = conventions.get_optimizations_path(split_path)
        if os.path.isdir(optimizations_dir):
            for entry in sorted(os.listdir(optimizations_dir)):
                results_path = conventions.get_optimization_results_path(split_path, entry)
                if os.path.exists(results_path):
                    optimizations[entry] = OptimizationResults.load(results_path)

        return cls(
            experiment_name=experiment_name,
            dataset_name=dataset_name,
            split=split,
            runs=runs,
            optimizations=optimizations,
        )
