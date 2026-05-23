"""
Evaluation metrics base class.

All metric implementations follow the same two-phase pattern:
  1. ``eval_sequence`` scores a single sequence and returns a results dict.
  2. ``combine_sequences`` aggregates per-sequence dicts into a dataset-level
     result, typically by summing count fields and recomputing derived rates.

Metric implementations are derived from the TrackEval library:
  https://github.com/JonathonLuiten/TrackEval
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np


class MetricBase(ABC):
    """Abstract base class for evaluation metrics."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used as a key in result dicts (e.g. 'HOTA')."""
        ...

    @property
    @abstractmethod
    def summary_fields(self) -> List[str]:
        """Field names included in summary tables and JSON output."""
        ...

    @abstractmethod
    def eval_sequence(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates a single preprocessed sequence.

        Args:
            data: Preprocessed sequence dict with keys 'gt_ids',
                'tracker_ids', 'similarity_scores', 'num_gt_ids', etc.

        Returns:
            Metric results dict whose keys are a superset of
            ``summary_fields``.
        """
        ...

    @abstractmethod
    def combine_sequences(self, all_res: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combines per-sequence results into a single dataset-level result.

        Args:
            all_res: Mapping from sequence name to the dict returned by
                ``eval_sequence``.

        Returns:
            Aggregated metric results dict.
        """
        ...

    @staticmethod
    def _combine_sum(all_res: Dict[str, Dict], field: str) -> Any:
        """Sums a field across all sequences."""
        return sum(all_res[k][field] for k in all_res)

    @staticmethod
    def _combine_weighted_av(
        all_res: Dict[str, Dict], field: str, comb_res: Dict, weight_field: str
    ) -> Any:
        """Weighted average of a field, using another field as the weight."""
        return sum(
            all_res[k][field] * all_res[k][weight_field] for k in all_res
        ) / np.maximum(1.0, comb_res[weight_field])
