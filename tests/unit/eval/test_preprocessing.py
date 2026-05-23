"""Tests for motrack.eval.preprocessing."""
import numpy as np

from motrack.eval.preprocessing import preprocess_sequence


def _make_raw_data(
    gt_ids, gt_dets, gt_classes, gt_zero_marked,
    tracker_ids, tracker_dets, tracker_confidences,
    similarity_scores, num_timesteps=1,
):
    """Helper to build a single-timestep raw_data dict."""
    return {
        'gt_ids': [np.array(gt_ids, dtype=int)],
        'gt_dets': [np.array(gt_dets, dtype=np.float32).reshape(-1, 4)],
        'gt_classes': [np.array(gt_classes, dtype=int)],
        'gt_extras': [{'zero_marked': np.array(gt_zero_marked, dtype=int)}],
        'tracker_ids': [np.array(tracker_ids, dtype=int)],
        'tracker_dets': [np.array(tracker_dets, dtype=np.float32).reshape(-1, 4)],
        'tracker_confidences': [np.array(tracker_confidences, dtype=np.float32)],
        'similarity_scores': [np.array(similarity_scores, dtype=np.float32)],
        'num_timesteps': num_timesteps,
    }


def test_distractor_removal():
    """Tracker dets matched to distractor GT should be removed."""
    raw = _make_raw_data(
        gt_ids=[1, 2],
        gt_dets=[[0, 0, 10, 10], [20, 20, 10, 10]],
        gt_classes=[1, 8],  # 1=pedestrian, 8=distractor
        gt_zero_marked=[1, 1],
        tracker_ids=[10, 20],
        tracker_dets=[[0, 0, 10, 10], [20, 20, 10, 10]],
        tracker_confidences=[0.9, 0.8],
        similarity_scores=[[1.0, 0.0], [0.0, 1.0]],  # perfect match
    )
    result = preprocess_sequence(raw, eval_classes={1}, distractor_classes={8})
    # Tracker det 20 matched to distractor GT 2 should be removed
    assert result['num_tracker_dets'] == 1
    # Only pedestrian GT kept
    assert result['num_gt_dets'] == 1


def test_zero_marked_filtering():
    """GT dets with zero_marked=0 should be excluded."""
    raw = _make_raw_data(
        gt_ids=[1, 2],
        gt_dets=[[0, 0, 10, 10], [20, 20, 10, 10]],
        gt_classes=[1, 1],
        gt_zero_marked=[1, 0],  # second is zero_marked
        tracker_ids=[10],
        tracker_dets=[[0, 0, 10, 10]],
        tracker_confidences=[0.9],
        similarity_scores=[[1.0], [0.0]],
    )
    result = preprocess_sequence(raw, eval_classes={1}, distractor_classes=set())
    assert result['num_gt_dets'] == 1
    assert result['num_gt_ids'] == 1


def test_id_relabeling():
    """IDs should be relabeled to contiguous 0-indexed."""
    raw = _make_raw_data(
        gt_ids=[5, 10],
        gt_dets=[[0, 0, 10, 10], [20, 20, 10, 10]],
        gt_classes=[1, 1],
        gt_zero_marked=[1, 1],
        tracker_ids=[100, 200],
        tracker_dets=[[0, 0, 10, 10], [20, 20, 10, 10]],
        tracker_confidences=[0.9, 0.8],
        similarity_scores=[[1.0, 0.0], [0.0, 1.0]],
    )
    result = preprocess_sequence(raw, eval_classes={1}, distractor_classes=set())
    np.testing.assert_array_equal(sorted(result['gt_ids'][0]), [0, 1])
    np.testing.assert_array_equal(sorted(result['tracker_ids'][0]), [0, 1])


def test_empty_sequence():
    """Empty GT and tracker should produce zero counts."""
    raw = _make_raw_data(
        gt_ids=[], gt_dets=[], gt_classes=[], gt_zero_marked=[],
        tracker_ids=[], tracker_dets=[], tracker_confidences=[],
        similarity_scores=np.empty((0, 0)),
    )
    result = preprocess_sequence(raw, eval_classes={1}, distractor_classes=set())
    assert result['num_gt_dets'] == 0
    assert result['num_tracker_dets'] == 0
    assert result['num_gt_ids'] == 0
    assert result['num_tracker_ids'] == 0
