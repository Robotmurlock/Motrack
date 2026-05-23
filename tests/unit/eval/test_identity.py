"""Tests for motrack.eval.metrics.identity."""
import numpy as np

from motrack.eval.metrics.identity import Identity


def _make_data(gt_ids, tracker_ids, similarity_scores):
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
        'num_timesteps': len(gt_ids),
        'num_gt_dets': all_gt,
        'num_tracker_dets': all_tr,
        'num_gt_ids': len(unique_gt),
        'num_tracker_ids': len(unique_tr),
    }


def test_perfect_identity():
    data = _make_data(
        gt_ids=[np.array([0, 1]), np.array([0, 1])],
        tracker_ids=[np.array([0, 1]), np.array([0, 1])],
        similarity_scores=[np.eye(2, dtype=np.float32), np.eye(2, dtype=np.float32)],
    )
    metric = Identity()
    res = metric.eval_sequence(data)
    np.testing.assert_allclose(res['IDF1'], 1.0)
    assert res['IDTP'] == 4
    assert res['IDFN'] == 0
    assert res['IDFP'] == 0


def test_no_tracker():
    data = _make_data(
        gt_ids=[np.array([0])],
        tracker_ids=[np.array([], dtype=int)],
        similarity_scores=[np.empty((1, 0), dtype=np.float32)],
    )
    metric = Identity()
    res = metric.eval_sequence(data)
    assert res['IDFN'] == 1
    assert res['IDTP'] == 0


def test_no_gt():
    data = _make_data(
        gt_ids=[np.array([], dtype=int)],
        tracker_ids=[np.array([0])],
        similarity_scores=[np.empty((0, 1), dtype=np.float32)],
    )
    metric = Identity()
    res = metric.eval_sequence(data)
    assert res['IDFP'] == 1
    assert res['IDTP'] == 0


def test_combine():
    metric = Identity()
    r1 = {'IDTP': 10, 'IDFN': 2, 'IDFP': 3, 'IDF1': 0, 'IDR': 0, 'IDP': 0}
    r2 = {'IDTP': 5, 'IDFN': 1, 'IDFP': 2, 'IDF1': 0, 'IDR': 0, 'IDP': 0}
    combined = metric.combine_sequences({'s1': r1, 's2': r2})
    assert combined['IDTP'] == 15
    assert combined['IDFN'] == 3
    assert combined['IDFP'] == 5
