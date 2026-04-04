"""Tests for motrack.eval.metrics.count."""
from motrack.eval.metrics.count import Count


def test_count_single_sequence():
    metric = Count()
    data = {'num_tracker_dets': 10, 'num_gt_dets': 12, 'num_tracker_ids': 3, 'num_gt_ids': 4}
    res = metric.eval_sequence(data)
    assert res['Dets'] == 10
    assert res['GT_Dets'] == 12
    assert res['IDs'] == 3
    assert res['GT_IDs'] == 4


def test_count_combine():
    metric = Count()
    all_res = {
        'seq1': {'Dets': 10, 'GT_Dets': 12, 'IDs': 3, 'GT_IDs': 4},
        'seq2': {'Dets': 20, 'GT_Dets': 25, 'IDs': 5, 'GT_IDs': 6},
    }
    combined = metric.combine_sequences(all_res)
    assert combined['Dets'] == 30
    assert combined['GT_Dets'] == 37
