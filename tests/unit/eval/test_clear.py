"""Tests for motrack.eval.metrics.clear."""
import numpy as np

from motrack.eval.metrics.clear import CLEAR


def _make_data(gt_ids, tracker_ids, similarity_scores, num_timesteps=None):
    """Build a minimal data dict for CLEAR metric."""
    if num_timesteps is None:
        num_timesteps = len(gt_ids)
    all_gt = sum(len(g) for g in gt_ids)
    all_tr = sum(len(t) for t in tracker_ids)
    unique_gt = set()
    unique_tr = set()
    for g in gt_ids:
        unique_gt.update(g.tolist())
    for t in tracker_ids:
        unique_tr.update(t.tolist())
    return {
        'gt_ids': gt_ids,
        'tracker_ids': tracker_ids,
        'similarity_scores': similarity_scores,
        'num_timesteps': num_timesteps,
        'num_gt_dets': all_gt,
        'num_tracker_dets': all_tr,
        'num_gt_ids': len(unique_gt),
        'num_tracker_ids': len(unique_tr),
    }


def test_perfect_tracking():
    """Perfect tracking: MOTA=1, no FP/FN/IDSW."""
    data = _make_data(
        gt_ids=[np.array([0, 1]), np.array([0, 1])],
        tracker_ids=[np.array([0, 1]), np.array([0, 1])],
        similarity_scores=[np.eye(2, dtype=np.float32), np.eye(2, dtype=np.float32)],
    )
    metric = CLEAR()
    res = metric.eval_sequence(data)
    assert res['CLR_TP'] == 4
    assert res['CLR_FP'] == 0
    assert res['CLR_FN'] == 0
    assert res['IDSW'] == 0
    np.testing.assert_allclose(res['MOTA'], 1.0)


def test_all_false_negatives():
    """No tracker output: all FN."""
    data = _make_data(
        gt_ids=[np.array([0, 1])],
        tracker_ids=[np.array([], dtype=int)],
        similarity_scores=[np.empty((2, 0), dtype=np.float32)],
    )
    metric = CLEAR()
    res = metric.eval_sequence(data)
    assert res['CLR_FN'] == 2
    assert res['CLR_TP'] == 0


def test_all_false_positives():
    """No GT: all FP."""
    data = _make_data(
        gt_ids=[np.array([], dtype=int)],
        tracker_ids=[np.array([0, 1])],
        similarity_scores=[np.empty((0, 2), dtype=np.float32)],
    )
    metric = CLEAR()
    res = metric.eval_sequence(data)
    assert res['CLR_FP'] == 2
    assert res['CLR_TP'] == 0


def test_id_switch():
    """Swapped IDs between frames should produce IDSW."""
    data = _make_data(
        gt_ids=[np.array([0, 1]), np.array([0, 1])],
        tracker_ids=[np.array([0, 1]), np.array([1, 0])],
        similarity_scores=[
            np.eye(2, dtype=np.float32),
            np.eye(2, dtype=np.float32),  # same IoU but swapped tracker IDs
        ],
    )
    metric = CLEAR()
    res = metric.eval_sequence(data)
    assert res['IDSW'] == 2  # both GT IDs get a different tracker ID
    assert res['CLR_TP'] == 4


def test_combine_sequences():
    metric = CLEAR()
    data1 = _make_data(
        gt_ids=[np.array([0])], tracker_ids=[np.array([0])],
        similarity_scores=[np.array([[0.8]], dtype=np.float32)],
    )
    data2 = _make_data(
        gt_ids=[np.array([0])], tracker_ids=[np.array([0])],
        similarity_scores=[np.array([[0.9]], dtype=np.float32)],
    )
    res1 = metric.eval_sequence(data1)
    res2 = metric.eval_sequence(data2)
    combined = metric.combine_sequences({'s1': res1, 's2': res2})
    assert combined['CLR_TP'] == 2
