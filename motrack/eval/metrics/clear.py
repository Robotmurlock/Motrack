"""
CLEAR MOT metrics (MOTA, MOTP, ID switches, track quality, etc.).

Frame-by-frame evaluation that matches detections with a priority for
maintaining ID continuity from the previous frame (1000x bonus in the
cost matrix), then falls back to IoU similarity.

See: https://link.springer.com/article/10.1007/s11263-007-0078-4

Derived from the TrackEval implementation:
  https://github.com/JonathonLuiten/TrackEval
"""
from typing import Any, Dict, List

import numpy as np
from scipy.optimize import linear_sum_assignment

from motrack.eval.metrics import MetricBase

DEFAULT_THRESHOLD = 0.5


class CLEAR(MetricBase):
    """
    CLEAR MOT metric.

    Per-sequence output fields:
      - Counts: CLR_TP, CLR_FN, CLR_FP, IDSW, Frag, CLR_Frames
      - Track quality: MT (mostly tracked >80%), PT (20-80%), ML (<20%)
      - Rates: MOTA, MOTP, MODA, CLR_Re, CLR_Pr, CLR_F1, sMOTA, MOTAL,
        MTR, PTR, MLR, FP_per_frame
      - Internal: MOTP_sum (used for combine_sequences)
    """

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold

    @property
    def name(self) -> str:
        return 'CLEAR'

    @property
    def integer_fields(self) -> List[str]:
        return ['CLR_TP', 'CLR_FN', 'CLR_FP', 'IDSW', 'MT', 'PT', 'ML', 'Frag', 'CLR_Frames']

    @property
    def float_fields(self) -> List[str]:
        return ['MOTA', 'MOTP', 'MODA', 'CLR_Re', 'CLR_Pr', 'MTR', 'PTR', 'MLR',
                'sMOTA', 'CLR_F1', 'FP_per_frame', 'MOTAL', 'MOTP_sum']

    @property
    def summed_fields(self) -> List[str]:
        return self.integer_fields + ['MOTP_sum']

    @property
    def summary_fields(self) -> List[str]:
        return ['MOTA', 'MOTP', 'MODA', 'CLR_Re', 'CLR_Pr', 'MTR', 'PTR', 'MLR', 'sMOTA',
                'CLR_TP', 'CLR_FN', 'CLR_FP', 'IDSW', 'MT', 'PT', 'ML', 'Frag']

    def eval_sequence(self, data: Dict[str, Any]) -> Dict[str, Any]:
        res = {f: 0 for f in self.integer_fields + self.float_fields}

        if data['num_tracker_dets'] == 0:
            res['CLR_FN'] = data['num_gt_dets']
            res['ML'] = data['num_gt_ids']
            res['MLR'] = 1.0
            return res
        if data['num_gt_dets'] == 0:
            res['CLR_FP'] = data['num_tracker_dets']
            res['MLR'] = 1.0
            return res

        num_gt_ids = data['num_gt_ids']
        gt_id_count = np.zeros(num_gt_ids)          # total frames each GT ID appears
        gt_matched_count = np.zeros(num_gt_ids)      # frames each GT ID is matched
        gt_frag_count = np.zeros(num_gt_ids)          # track-start transitions per GT

        # prev_tracker_id: last tracker ID assigned to each GT (persists across gaps)
        # prev_timestep_tracker_id: tracker ID from the immediately previous frame only
        prev_tracker_id = np.nan * np.zeros(num_gt_ids)
        prev_timestep_tracker_id = np.nan * np.zeros(num_gt_ids)

        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            if len(gt_ids_t) == 0:
                res['CLR_FP'] += len(tracker_ids_t)
                continue
            if len(tracker_ids_t) == 0:
                res['CLR_FN'] += len(gt_ids_t)
                gt_id_count[gt_ids_t] += 1
                continue

            similarity = data['similarity_scores'][t]

            # Build cost matrix: 1000x bonus for continuing the same tracker
            # ID from the previous frame, plus the IoU similarity. Pairs
            # below the IoU threshold are zeroed out to prevent matching.
            score_mat = (tracker_ids_t[np.newaxis, :] == prev_timestep_tracker_id[gt_ids_t[:, np.newaxis]])
            score_mat = 1000 * score_mat + similarity
            score_mat[similarity < self.threshold - np.finfo('float').eps] = 0

            match_rows, match_cols = linear_sum_assignment(-score_mat)
            actually_matched_mask = score_mat[match_rows, match_cols] > 0 + np.finfo('float').eps
            match_rows = match_rows[actually_matched_mask]
            match_cols = match_cols[actually_matched_mask]

            matched_gt_ids = gt_ids_t[match_rows]
            matched_tracker_ids = tracker_ids_t[match_cols]

            # Count ID switches: GT was previously matched to a different tracker ID
            prev_matched_tracker_ids = prev_tracker_id[matched_gt_ids]
            is_idsw = np.logical_not(np.isnan(prev_matched_tracker_ids)) & \
                      np.not_equal(matched_tracker_ids, prev_matched_tracker_ids)
            res['IDSW'] += np.sum(is_idsw)

            # Update per-ID counters for MT/PT/ML/Frag
            gt_id_count[gt_ids_t] += 1
            gt_matched_count[matched_gt_ids] += 1
            not_previously_tracked = np.isnan(prev_timestep_tracker_id)
            prev_tracker_id[matched_gt_ids] = matched_tracker_ids
            prev_timestep_tracker_id[:] = np.nan
            prev_timestep_tracker_id[matched_gt_ids] = matched_tracker_ids
            currently_tracked = np.logical_not(np.isnan(prev_timestep_tracker_id))
            gt_frag_count += np.logical_and(not_previously_tracked, currently_tracked)

            num_matches = len(matched_gt_ids)
            res['CLR_TP'] += num_matches
            res['CLR_FN'] += len(gt_ids_t) - num_matches
            res['CLR_FP'] += len(tracker_ids_t) - num_matches
            if num_matches > 0:
                res['MOTP_sum'] += sum(similarity[match_rows, match_cols])

        # Track quality: MT (>80% matched), PT (20-80%), ML (<20%)
        tracked_ratio = gt_matched_count[gt_id_count > 0] / gt_id_count[gt_id_count > 0]
        res['MT'] = int(np.sum(np.greater(tracked_ratio, 0.8)))
        res['PT'] = int(np.sum(np.greater_equal(tracked_ratio, 0.2))) - res['MT']
        res['ML'] = num_gt_ids - res['MT'] - res['PT']
        res['Frag'] = int(np.sum(np.subtract(gt_frag_count[gt_frag_count > 0], 1)))
        res['MOTP'] = res['MOTP_sum'] / np.maximum(1.0, res['CLR_TP'])
        res['CLR_Frames'] = data['num_timesteps']

        res = self._compute_final_fields(res)
        return res

    def combine_sequences(self, all_res: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        res = {}
        for field in self.summed_fields:
            res[field] = self._combine_sum(all_res, field)
        res = self._compute_final_fields(res)
        return res

    @staticmethod
    def _compute_final_fields(res: Dict[str, Any]) -> Dict[str, Any]:
        """Derives rate metrics from accumulated counts."""
        num_gt_ids = res['MT'] + res['ML'] + res['PT']
        res['MTR'] = res['MT'] / np.maximum(1.0, num_gt_ids)
        res['MLR'] = res['ML'] / np.maximum(1.0, num_gt_ids)
        res['PTR'] = res['PT'] / np.maximum(1.0, num_gt_ids)
        res['CLR_Re'] = res['CLR_TP'] / np.maximum(1.0, res['CLR_TP'] + res['CLR_FN'])
        res['CLR_Pr'] = res['CLR_TP'] / np.maximum(1.0, res['CLR_TP'] + res['CLR_FP'])
        res['MODA'] = (res['CLR_TP'] - res['CLR_FP']) / np.maximum(1.0, res['CLR_TP'] + res['CLR_FN'])
        res['MOTA'] = (res['CLR_TP'] - res['CLR_FP'] - res['IDSW']) / np.maximum(1.0, res['CLR_TP'] + res['CLR_FN'])
        res['MOTP'] = res['MOTP_sum'] / np.maximum(1.0, res['CLR_TP'])
        res['sMOTA'] = (res['MOTP_sum'] - res['CLR_FP'] - res['IDSW']) / np.maximum(1.0, res['CLR_TP'] + res['CLR_FN'])
        res['CLR_F1'] = res['CLR_TP'] / np.maximum(1.0, res['CLR_TP'] + 0.5 * res['CLR_FN'] + 0.5 * res['CLR_FP'])
        res['FP_per_frame'] = res['CLR_FP'] / np.maximum(1.0, res['CLR_Frames'])
        safe_log_idsw = np.log10(res['IDSW']) if res['IDSW'] > 0 else res['IDSW']
        res['MOTAL'] = (res['CLR_TP'] - res['CLR_FP'] - safe_log_idsw) / np.maximum(1.0, res['CLR_TP'] + res['CLR_FN'])
        return res
