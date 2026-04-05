"""
Collectors that aggregate all tool outputs for an experiment.
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional

from motrack.common import conventions
from tools.data.eval import EvalResults
from tools.data.optimization import OptimizationResults
from tools.data.inference import InferenceOutputData


@dataclass
class TrackerRunResult:
    """Complete result for a single tracker run (one config-hash directory)."""
    config_hash: str
    inference_output: InferenceOutputData
    eval_results: Optional[EvalResults] = None

    @classmethod
    def load(cls, run_dir: str) -> 'TrackerRunResult':
        """Load all artifacts from a config-hash directory."""
        config_hash = os.path.basename(run_dir)

        inference_output = InferenceOutputData.load(conventions.get_run_meta_path(run_dir))

        eval_path = conventions.get_eval_results_path(run_dir)
        eval_results = EvalResults.load(eval_path) if os.path.exists(eval_path) else None

        return cls(
            config_hash=config_hash,
            inference_output=inference_output,
            eval_results=eval_results,
        )


@dataclass
class ExperimentResults:
    """All runs under an experiment/split, with optional optimization results.

    This is the top-level collector the Streamlit frontend will consume.
    """
    experiment_name: str
    dataset_name: str
    split: str
    runs: List[TrackerRunResult] = field(default_factory=list)
    optimization: Optional[OptimizationResults] = None

    @classmethod
    def collect(cls, split_path: str) -> 'ExperimentResults':
        """
        Walk a split directory and load all tracker run results.

        Args:
            split_path: ``{master}/{dataset}/{experiment}/{split}/``

        Returns:
            Populated ``ExperimentResults`` with all runs and optional
            optimization results.
        """
        split = os.path.basename(split_path)
        experiment_name = os.path.basename(os.path.dirname(split_path))
        dataset_name = os.path.basename(os.path.dirname(os.path.dirname(split_path)))

        runs: List[TrackerRunResult] = []
        for entry in sorted(os.listdir(split_path)):
            entry_path = os.path.join(split_path, entry)
            if not os.path.isdir(entry_path):
                continue
            inference_output_path = conventions.get_run_meta_path(entry_path)
            if not os.path.exists(inference_output_path):
                continue
            runs.append(TrackerRunResult.load(entry_path))

        optim_path = conventions.get_optimization_results_path(split_path)
        optimization = OptimizationResults.load(optim_path) if os.path.exists(optim_path) else None

        return cls(
            experiment_name=experiment_name,
            dataset_name=dataset_name,
            split=split,
            runs=runs,
            optimization=optimization,
        )
