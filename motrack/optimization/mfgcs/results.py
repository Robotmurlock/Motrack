"""
Per-coordinate / per-sweep history records for MFGCS runs.

These structures live in ``OptimizationResults.extras['mfgcs_history']`` and
preserve a full audit trail of the sweep: which subset was sampled, what
candidate the coordinate optimizer chose, what its low- and high-fidelity
scores were, and whether the move was accepted.
"""
import dataclasses
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class MFGCSCoordinateRecord:
    """One coordinate move within a sweep."""
    sweep: int
    coord_index: int
    dotpath: str
    previous_value: Any
    candidate_value: Any
    accepted: bool
    low_score: Optional[float]
    full_score: Optional[float]
    sampled_scenes: List[str] = field(default_factory=list)
    skipped_full_eval: bool = False  # candidate == previous, no full-fidelity check
    note: str = ''  # human-readable info (e.g. "degenerate window")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class MFGCSSweepRecord:
    """One full sweep over all parameters."""
    sweep: int
    accepted_count: int
    coordinates: List[MFGCSCoordinateRecord]

    def to_dict(self) -> dict:
        return {
            'sweep': self.sweep,
            'accepted_count': self.accepted_count,
            'coordinates': [c.to_dict() for c in self.coordinates],
        }
