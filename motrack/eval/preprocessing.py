"""
Evaluation data preprocessing.

Filters ground-truth and tracker detections for a single sequence before
metric computation. Configurable eval/distractor classes allow reuse across
datasets beyond MOT Challenge.
"""
from typing import Any, Dict, Set

import numpy as np
from scipy.optimize import linear_sum_assignment

from motrack.eval.similarity import compute_box_ious

DISTRACTOR_IOU_THRESHOLD = 0.5


def preprocess_sequence(
    raw_data: Dict[str, Any],
    eval_classes: Set[int],
    distractor_classes: Set[int],
) -> Dict[str, Any]:
    """
    Preprocesses raw sequence data for metric evaluation.

    Steps per timestep:
      1. Match tracker dets to distractor GT via Hungarian at IoU >= 0.5,
         remove matched tracker dets.
      2. Keep only GT where class_id is in eval_classes AND zero_marked != 0.
      3. Relabel GT and tracker IDs to contiguous 0-indexed integers.

    Args:
        raw_data: Dict produced by merging load_mot_gt and load_mot_tracker
                  results, plus 'similarity_scores' per timestep.
        eval_classes: Set of class IDs to evaluate.
        distractor_classes: Set of class IDs treated as distractors.

    Returns:
        Preprocessed data dict ready for metric computation.
    """
    num_timesteps = raw_data['num_timesteps']
    data = {
        'gt_ids': [None] * num_timesteps,
        'tracker_ids': [None] * num_timesteps,
        'gt_dets': [None] * num_timesteps,
        'tracker_dets': [None] * num_timesteps,
        'tracker_confidences': [None] * num_timesteps,
        'similarity_scores': [None] * num_timesteps,
    }

    unique_gt_ids = []
    unique_tracker_ids = []
    num_gt_dets = 0
    num_tracker_dets = 0

    for t in range(num_timesteps):
        gt_ids = raw_data['gt_ids'][t]
        gt_dets = raw_data['gt_dets'][t]
        gt_classes = raw_data['gt_classes'][t]
        gt_zero_marked = raw_data['gt_extras'][t]['zero_marked']

        tracker_ids = raw_data['tracker_ids'][t]
        tracker_dets = raw_data['tracker_dets'][t]
        tracker_confidences = raw_data['tracker_confidences'][t]
        similarity_scores = raw_data['similarity_scores'][t]

        # Step 1: Remove tracker dets matched to distractor GT
        to_remove_tracker = np.array([], dtype=np.int32)
        if len(gt_ids) > 0 and len(tracker_ids) > 0:
            matching_scores = similarity_scores.copy()
            matching_scores[matching_scores < DISTRACTOR_IOU_THRESHOLD - np.finfo('float').eps] = 0
            match_rows, match_cols = linear_sum_assignment(-matching_scores)
            actually_matched = matching_scores[match_rows, match_cols] > 0 + np.finfo('float').eps
            match_rows = match_rows[actually_matched]
            match_cols = match_cols[actually_matched]

            is_distractor = np.isin(gt_classes[match_rows], list(distractor_classes))
            to_remove_tracker = match_cols[is_distractor]

        tracker_ids = np.delete(tracker_ids, to_remove_tracker, axis=0)
        tracker_dets = np.delete(tracker_dets, to_remove_tracker, axis=0)
        tracker_confidences = np.delete(tracker_confidences, to_remove_tracker, axis=0)
        similarity_scores = np.delete(similarity_scores, to_remove_tracker, axis=1)

        # Step 2: Keep only eval-class GT with zero_marked != 0
        gt_keep_mask = np.isin(gt_classes, list(eval_classes)) & (gt_zero_marked != 0)
        gt_ids = gt_ids[gt_keep_mask]
        gt_dets = gt_dets[gt_keep_mask]
        similarity_scores = similarity_scores[gt_keep_mask]

        data['gt_ids'][t] = gt_ids
        data['gt_dets'][t] = gt_dets
        data['tracker_ids'][t] = tracker_ids
        data['tracker_dets'][t] = tracker_dets
        data['tracker_confidences'][t] = tracker_confidences
        data['similarity_scores'][t] = similarity_scores

        unique_gt_ids += list(np.unique(gt_ids))
        unique_tracker_ids += list(np.unique(tracker_ids))
        num_gt_dets += len(gt_ids)
        num_tracker_dets += len(tracker_ids)

    # Step 3: Relabel IDs to contiguous 0-indexed
    if len(unique_gt_ids) > 0:
        unique_gt_ids = np.unique(unique_gt_ids)
        gt_id_map = np.full(int(np.max(unique_gt_ids)) + 1, fill_value=-1, dtype=np.int32)
        gt_id_map[unique_gt_ids] = np.arange(len(unique_gt_ids))
        for t in range(num_timesteps):
            if len(data['gt_ids'][t]) > 0:
                data['gt_ids'][t] = gt_id_map[data['gt_ids'][t]].astype(np.int32)
    else:
        unique_gt_ids = np.array([], dtype=int)

    if len(unique_tracker_ids) > 0:
        unique_tracker_ids = np.unique(unique_tracker_ids)
        tracker_id_map = np.full(int(np.max(unique_tracker_ids)) + 1, fill_value=-1, dtype=np.int32)
        tracker_id_map[unique_tracker_ids] = np.arange(len(unique_tracker_ids))
        for t in range(num_timesteps):
            if len(data['tracker_ids'][t]) > 0:
                data['tracker_ids'][t] = tracker_id_map[data['tracker_ids'][t]].astype(np.int32)
    else:
        unique_tracker_ids = np.array([], dtype=int)

    data['num_gt_ids'] = len(unique_gt_ids)
    data['num_tracker_ids'] = len(unique_tracker_ids)
    data['num_gt_dets'] = num_gt_dets
    data['num_tracker_dets'] = num_tracker_dets
    data['num_timesteps'] = num_timesteps

    return data
