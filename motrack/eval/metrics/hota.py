"""
HOTA (Higher Order Tracking Accuracy) metric.

Evaluates tracking quality across multiple IoU thresholds (alpha) by
decomposing performance into detection accuracy (DetA) and association
accuracy (AssA).  HOTA = sqrt(DetA * AssA).

See: https://link.springer.com/article/10.1007/s11263-020-01375-2

Derived from the TrackEval implementation:
  https://github.com/JonathonLuiten/TrackEval
"""
from typing import Any, Dict, List

import numpy as np
from scipy.optimize import linear_sum_assignment

from motrack.eval.metrics import MetricBase

# 19 IoU thresholds from 0.05 to 0.95 in steps of 0.05.
# Each threshold defines a different strictness level for what counts as
# a true-positive detection match. Final scores are averaged over all alphas.
ALPHA_THRESHOLDS = np.arange(0.05, 0.99, 0.05)


class HOTA(MetricBase):
    """
    HOTA metric with per-alpha detection and association sub-metrics.

    Per-sequence output fields:
      - Array fields (one value per alpha):
        HOTA, DetA, AssA, DetRe, DetPr, AssRe, AssPr, LocA, OWTA,
        HOTA_TP, HOTA_FN, HOTA_FP
      - Scalar fields (alpha=0.05 snapshot):
        HOTA(0), LocA(0), HOTALocA(0)
    """

    @property
    def name(self) -> str:
        return 'HOTA'

    @property
    def integer_array_fields(self) -> List[str]:
        return ['HOTA_TP', 'HOTA_FN', 'HOTA_FP']

    @property
    def float_array_fields(self) -> List[str]:
        return ['HOTA', 'DetA', 'AssA', 'DetRe', 'DetPr', 'AssRe', 'AssPr', 'LocA', 'OWTA']

    @property
    def float_fields(self) -> List[str]:
        return ['HOTA(0)', 'LocA(0)', 'HOTALocA(0)']

    @property
    def summary_fields(self) -> List[str]:
        return self.float_array_fields + self.float_fields

    def eval_sequence(self, data: Dict[str, Any]) -> Dict[str, Any]:
        n_alpha = len(ALPHA_THRESHOLDS)
        res = {}
        for field in self.float_array_fields + self.integer_array_fields:
            res[field] = np.zeros(n_alpha, dtype=np.float32)
        for field in self.float_fields:
            res[field] = 0

        # Early return for degenerate cases
        if data['num_tracker_dets'] == 0:
            res['HOTA_FN'] = data['num_gt_dets'] * np.ones(n_alpha, dtype=np.float32)
            res['LocA'] = np.ones(n_alpha, dtype=np.float32)
            res['LocA(0)'] = 1.0
            return res
        if data['num_gt_dets'] == 0:
            res['HOTA_FP'] = data['num_tracker_dets'] * np.ones(n_alpha, dtype=np.float32)
            res['LocA'] = np.ones(n_alpha, dtype=np.float32)
            res['LocA(0)'] = 1.0
            return res

        # ------------------------------------------------------------------
        # Pass 1: Accumulate global alignment scores between ID pairs.
        #
        # For each (gt_id, tracker_id) pair, we sum a normalized similarity
        # (Jaccard-style IoU between the per-frame similarity vectors) across
        # all timesteps. This captures how consistently a tracker ID follows
        # a GT ID across the whole sequence, independent of the alpha
        # threshold. The resulting `global_alignment_score` is used in pass 2
        # to bias the Hungarian matching toward globally consistent ID pairs.
        # ------------------------------------------------------------------
        potential_matches_count = np.zeros((data['num_gt_ids'], data['num_tracker_ids']))
        gt_id_count = np.zeros((data['num_gt_ids'], 1))
        tracker_id_count = np.zeros((1, data['num_tracker_ids']))

        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            similarity = data['similarity_scores'][t]
            # Normalize per-frame similarity into a Jaccard-style score so
            # that each (gt, tracker) pair's contribution is in [0, 1].
            sim_iou_denom = similarity.sum(0)[np.newaxis, :] + similarity.sum(1)[:, np.newaxis] - similarity
            sim_iou = np.zeros_like(similarity)
            sim_iou_mask = sim_iou_denom > 0 + np.finfo('float').eps
            sim_iou[sim_iou_mask] = similarity[sim_iou_mask] / sim_iou_denom[sim_iou_mask]

            potential_matches_count[gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]] += sim_iou
            gt_id_count[gt_ids_t] += 1
            tracker_id_count[0, tracker_ids_t] += 1

        # Jaccard alignment: intersection / union of ID occurrence counts
        global_alignment_score = potential_matches_count / (gt_id_count + tracker_id_count - potential_matches_count)
        matches_counts = [np.zeros_like(potential_matches_count) for _ in ALPHA_THRESHOLDS]

        # ------------------------------------------------------------------
        # Pass 2: Per-timestep, per-alpha matching.
        #
        # At each timestep the score matrix combines per-frame IoU with the
        # global alignment score so that the Hungarian algorithm prefers
        # assignments that are both locally accurate and globally consistent.
        # After matching, detections with IoU >= alpha are counted as TP.
        # ------------------------------------------------------------------
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            if len(gt_ids_t) == 0:
                for a in range(n_alpha):
                    res['HOTA_FP'][a] += len(tracker_ids_t)
                continue
            if len(tracker_ids_t) == 0:
                for a in range(n_alpha):
                    res['HOTA_FN'][a] += len(gt_ids_t)
                continue

            similarity = data['similarity_scores'][t]
            score_mat = global_alignment_score[gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]] * similarity
            match_rows, match_cols = linear_sum_assignment(-score_mat)

            for a, alpha in enumerate(ALPHA_THRESHOLDS):
                matched_mask = similarity[match_rows, match_cols] >= alpha - np.finfo('float').eps
                alpha_match_rows = match_rows[matched_mask]
                alpha_match_cols = match_cols[matched_mask]
                num_matches = len(alpha_match_rows)

                res['HOTA_TP'][a] += num_matches
                res['HOTA_FN'][a] += len(gt_ids_t) - num_matches
                res['HOTA_FP'][a] += len(tracker_ids_t) - num_matches
                if num_matches > 0:
                    res['LocA'][a] += sum(similarity[alpha_match_rows, alpha_match_cols])
                    matches_counts[a][gt_ids_t[alpha_match_rows], tracker_ids_t[alpha_match_cols]] += 1

        # ------------------------------------------------------------------
        # Compute association scores (AssA, AssRe, AssPr) per alpha.
        #
        # For each (gt_id, tracker_id) pair, association accuracy is the
        # Jaccard index of their matched frame counts vs total frame counts.
        # The per-pair scores are then TP-weighted averaged.
        # ------------------------------------------------------------------
        for a in range(n_alpha):
            mc = matches_counts[a]
            ass_a = mc / np.maximum(1, gt_id_count + tracker_id_count - mc)
            res['AssA'][a] = np.sum(mc * ass_a) / np.maximum(1, res['HOTA_TP'][a])
            ass_re = mc / np.maximum(1, gt_id_count)
            res['AssRe'][a] = np.sum(mc * ass_re) / np.maximum(1, res['HOTA_TP'][a])
            ass_pr = mc / np.maximum(1, tracker_id_count)
            res['AssPr'][a] = np.sum(mc * ass_pr) / np.maximum(1, res['HOTA_TP'][a])

        # Average localization accuracy over matched TPs
        res['LocA'] = np.maximum(1e-10, res['LocA']) / np.maximum(1e-10, res['HOTA_TP'])
        res = self._compute_final_fields(res)
        return res

    def combine_sequences(self, all_res: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        res = {}
        for field in self.integer_array_fields:
            res[field] = self._combine_sum(all_res, field)
        for field in ['AssRe', 'AssPr', 'AssA']:
            res[field] = self._combine_weighted_av(all_res, field, res, weight_field='HOTA_TP')
        loca_weighted_sum = sum(all_res[k]['LocA'] * all_res[k]['HOTA_TP'] for k in all_res)
        res['LocA'] = np.maximum(1e-10, loca_weighted_sum) / np.maximum(1e-10, res['HOTA_TP'])
        res = self._compute_final_fields(res)
        return res

    @staticmethod
    def _compute_final_fields(res: Dict[str, Any]) -> Dict[str, Any]:
        """Derives detection and HOTA scores from TP/FN/FP counts."""
        res['DetRe'] = res['HOTA_TP'] / np.maximum(1, res['HOTA_TP'] + res['HOTA_FN'])
        res['DetPr'] = res['HOTA_TP'] / np.maximum(1, res['HOTA_TP'] + res['HOTA_FP'])
        res['DetA'] = res['HOTA_TP'] / np.maximum(1, res['HOTA_TP'] + res['HOTA_FN'] + res['HOTA_FP'])
        res['HOTA'] = np.sqrt(res['DetA'] * res['AssA'])
        res['OWTA'] = np.sqrt(res['DetRe'] * res['AssA'])
        res['HOTA(0)'] = res['HOTA'][0]
        res['LocA(0)'] = res['LocA'][0]
        res['HOTALocA(0)'] = res['HOTA(0)'] * res['LocA(0)']
        return res
