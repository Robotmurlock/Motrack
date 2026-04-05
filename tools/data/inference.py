"""
Typed structure for ``inference output data (``run_meta.json``)``.
"""
import dataclasses
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class OptunaOutputData:
    """Optuna trial metadata attached to a tracker run."""
    study_name: str
    trial_number: int
    trial_params: Dict[str, Any]


@dataclass
class InferenceOutputData:
    """Metadata for a single tracker run (``inference output data (``run_meta.json``)``)."""
    created_at: str
    optuna: Optional[OptunaOutputData] = None

    def to_dict(self) -> dict:
        d: dict = {'created_at': self.created_at}
        if self.optuna is not None:
            d['optuna'] = dataclasses.asdict(self.optuna)
        return d

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'InferenceOutputData':
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        optuna_raw = raw.pop('optuna', None)
        optuna = OptunaOutputData(**optuna_raw) if optuna_raw is not None else None
        return cls(created_at=raw['created_at'], optuna=optuna)
