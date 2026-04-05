"""
Typed structure for ``eval_results.json``.
"""
import dataclasses
import json
import os
from dataclasses import dataclass
from typing import Dict


@dataclass
class EvalResults:
    """Evaluation metrics for a single tracker run (``eval_results.json``).

    Metric fields vary per metric type (HOTA, CLEAR, Identity, Count) so the
    inner dicts remain untyped.
    """
    combined: Dict[str, Dict[str, float]]
    sequences: Dict[str, Dict[str, Dict[str, float]]]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'EvalResults':
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        return cls(combined=raw['combined'], sequences=raw['sequences'])
