"""
Typed structure for ``optimization_results.json``.
"""
import dataclasses
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class TrialResult:
    """Summary of a single Optuna trial."""
    number: int
    value: float
    params: Dict[str, Any]
    state: str


@dataclass
class OptimizationResults:
    """Aggregated optimization results (``optimization_results.json``).

    Stored at the split level because it spans multiple config-hash
    directories (one per trial).
    """
    study_name: str
    best_trial: TrialResult
    all_trials: List[TrialResult]

    def to_dict(self) -> dict:
        return {
            'study_name': self.study_name,
            'best_trial': dataclasses.asdict(self.best_trial),
            'all_trials': [dataclasses.asdict(t) for t in self.all_trials],
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'OptimizationResults':
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        return cls(
            study_name=raw['study_name'],
            best_trial=TrialResult(**raw['best_trial']),
            all_trials=[TrialResult(**t) for t in raw['all_trials']],
        )
