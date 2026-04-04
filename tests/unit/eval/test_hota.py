"""Tests for motrack.eval.metrics.hota."""
import numpy as np

from motrack.eval.metrics.hota import HOTA


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


def test_perfect_tracking():
    """Perfect 1:1 matching should give high HOTA."""
    data = _make_data(
        gt_ids=[np.array([0, 1]), np.array([0, 1])],
        tracker_ids=[np.array([0, 1]), np.array([0, 1])],
        similarity_scores=[np.eye(2, dtype=np.float32), np.eye(2, dtype=np.float32)],
    )
    metric = HOTA()
    res = metric.eval_sequence(data)
    # Perfect tracking: DetA=1, AssA=1, HOTA=1 for all alpha <= 1.0
    np.testing.assert_allclose(res['DetA'], 1.0, atol=1e-6)
    np.testing.assert_allclose(res['AssA'], 1.0, atol=1e-6)
    np.testing.assert_allclose(res['HOTA'], 1.0, atol=1e-6)


def test_no_tracker_dets():
    data = _make_data(
        gt_ids=[np.array([0])],
        tracker_ids=[np.array([], dtype=int)],
        similarity_scores=[np.empty((1, 0), dtype=np.float32)],
    )
    metric = HOTA()
    res = metric.eval_sequence(data)
    np.testing.assert_array_equal(res['HOTA_FN'], np.ones(19))
    np.testing.assert_array_equal(res['HOTA_TP'], np.zeros(19))


def test_no_gt_dets():
    data = _make_data(
        gt_ids=[np.array([], dtype=int)],
        tracker_ids=[np.array([0])],
        similarity_scores=[np.empty((0, 1), dtype=np.float32)],
    )
    metric = HOTA()
    res = metric.eval_sequence(data)
    np.testing.assert_array_equal(res['HOTA_FP'], np.ones(19))
    np.testing.assert_array_equal(res['HOTA_TP'], np.zeros(19))


def test_combine_sequences():
    metric = HOTA()
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
    # Combined TP should be sum of individual TPs
    np.testing.assert_array_equal(combined['HOTA_TP'], res1['HOTA_TP'] + res2['HOTA_TP'])
