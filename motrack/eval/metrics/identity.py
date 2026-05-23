"""
Identity metrics (IDF1, IDR, IDP).

Measures how well tracker IDs correspond to GT IDs across an entire
sequence via a global optimal assignment. Builds a padded cost matrix
of size (N+M) x (N+M) where N=num_gt_ids, M=num_tracker_ids, and solves
with the Hungarian algorithm.

See: https://arxiv.org/abs/1609.01775

Derived from the TrackEval implementation:
  https://github.com/JonathonLuiten/TrackEval
"""
from typing import Any, Dict, List

import numpy as np
from scipy.optimize import linear_sum_assignment

from motrack.eval.metrics import MetricBase

DEFAULT_THRESHOLD = 0.5


class Identity(MetricBase):
    """
    Identity metric for global ID consistency.

    Per-sequence output fields:
      - IDF1: harmonic mean of IDR and IDP
      - IDR: ID recall (IDTP / (IDTP + IDFN))
      - IDP: ID precision (IDTP / (IDTP + IDFP))
      - IDTP, IDFN, IDFP: integer counts
    """

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold

    @property
    def name(self) -> str:
        return 'Identity'

    @property
    def summary_fields(self) -> List[str]:
        return ['IDF1', 'IDR', 'IDP', 'IDTP', 'IDFN', 'IDFP']

    def eval_sequence(self, data: Dict[str, Any]) -> Dict[str, Any]:
        res = {f: 0 for f in self.summary_fields}

        if data['num_tracker_dets'] == 0:
            res['IDFN'] = data['num_gt_dets']
            return res
        if data['num_gt_dets'] == 0:
            res['IDFP'] = data['num_tracker_dets']
            return res

        # Count frames where each (gt_id, tracker_id) pair co-occurs above
        # the IoU threshold, and total frames per ID.
        potential_matches_count = np.zeros((data['num_gt_ids'], data['num_tracker_ids']))
        gt_id_count = np.zeros(data['num_gt_ids'])
        tracker_id_count = np.zeros(data['num_tracker_ids'])

        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            matches_mask = np.greater_equal(data['similarity_scores'][t], self.threshold)
            match_idx_gt, match_idx_tracker = np.nonzero(matches_mask)
            potential_matches_count[gt_ids_t[match_idx_gt], tracker_ids_t[match_idx_tracker]] += 1
            gt_id_count[gt_ids_t] += 1
            tracker_id_count[tracker_ids_t] += 1

        # Build a padded (N+M) x (N+M) cost matrix.
        # Top-left NxM block: real assignment costs (FN + FP per pair).
        # Bottom-left MxM block: cost of leaving a tracker ID unassigned (FP).
        # Top-right NxN block: cost of leaving a GT ID unassigned (FN).
        # The padding with 1e10 prevents invalid cross-assignments.
        num_gt_ids = data['num_gt_ids']
        num_tracker_ids = data['num_tracker_ids']
        n = num_gt_ids + num_tracker_ids

        fp_mat = np.zeros((n, n))
        fn_mat = np.zeros((n, n))
        fp_mat[num_gt_ids:, :num_tracker_ids] = 1e10
        fn_mat[:num_gt_ids, num_tracker_ids:] = 1e10

        for gt_id in range(num_gt_ids):
            fn_mat[gt_id, :num_tracker_ids] = gt_id_count[gt_id]
            fn_mat[gt_id, num_tracker_ids + gt_id] = gt_id_count[gt_id]
        for tracker_id in range(num_tracker_ids):
            fp_mat[:num_gt_ids, tracker_id] = tracker_id_count[tracker_id]
            fp_mat[tracker_id + num_gt_ids, tracker_id] = tracker_id_count[tracker_id]

        fn_mat[:num_gt_ids, :num_tracker_ids] -= potential_matches_count
        fp_mat[:num_gt_ids, :num_tracker_ids] -= potential_matches_count

        match_rows, match_cols = linear_sum_assignment(fn_mat + fp_mat)

        res['IDFN'] = int(fn_mat[match_rows, match_cols].sum())
        res['IDFP'] = int(fp_mat[match_rows, match_cols].sum())
        res['IDTP'] = int(gt_id_count.sum()) - res['IDFN']

        res = self._compute_final_fields(res)
        return res

    def combine_sequences(self, all_res: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        res = {}
        for field in ['IDTP', 'IDFN', 'IDFP']:
            res[field] = self._combine_sum(all_res, field)
        res = self._compute_final_fields(res)
        return res

    @staticmethod
    def _compute_final_fields(res: Dict[str, Any]) -> Dict[str, Any]:
        """Derives ID recall, precision, and F1 from counts."""
        res['IDR'] = res['IDTP'] / np.maximum(1.0, res['IDTP'] + res['IDFN'])
        res['IDP'] = res['IDTP'] / np.maximum(1.0, res['IDTP'] + res['IDFP'])
        res['IDF1'] = res['IDTP'] / np.maximum(1.0, res['IDTP'] + 0.5 * res['IDFP'] + 0.5 * res['IDFN'])
        return res
