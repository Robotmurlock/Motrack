"""
Count metric — simple detection and ID tallies.

Derived from the TrackEval implementation:
  https://github.com/JonathonLuiten/TrackEval
"""
from typing import Any, Dict, List

from motrack.eval.metrics import MetricBase


class Count(MetricBase):
    """
    Counts detections and unique IDs per sequence.

    Per-sequence output fields:
      - Dets: total tracker detections
      - GT_Dets: total ground-truth detections
      - IDs: unique tracker track IDs
      - GT_IDs: unique ground-truth track IDs
    """

    @property
    def name(self) -> str:
        return 'Count'

    @property
    def summary_fields(self) -> List[str]:
        return ['Dets', 'GT_Dets', 'IDs', 'GT_IDs']

    def eval_sequence(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'Dets': data['num_tracker_dets'],
            'GT_Dets': data['num_gt_dets'],
            'IDs': data['num_tracker_ids'],
            'GT_IDs': data['num_gt_ids'],
        }

    def combine_sequences(self, all_res: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        return {field: self._combine_sum(all_res, field) for field in self.summary_fields}
