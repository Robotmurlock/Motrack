"""Tests for motrack.eval.io."""
import numpy as np

from motrack.eval.io import load_mot_gt, load_mot_tracker


def test_load_gt(tmp_path):
    gt_file = tmp_path / "gt.txt"
    gt_file.write_text(
        "1,1,100,200,50,60,1,1,0.9\n"
        "1,2,300,400,70,80,1,1,0.8\n"
        "2,1,105,205,50,60,1,1,0.9\n"
    )
    result = load_mot_gt(str(gt_file), num_timesteps=3)
    assert len(result['gt_ids']) == 3

    # Frame 1: two detections
    np.testing.assert_array_equal(result['gt_ids'][0], [1, 2])
    np.testing.assert_allclose(result['gt_dets'][0][0], [100, 200, 50, 60])
    np.testing.assert_array_equal(result['gt_classes'][0], [1, 1])
    np.testing.assert_array_equal(result['gt_extras'][0]['zero_marked'], [1, 1])

    # Frame 2: one detection
    np.testing.assert_array_equal(result['gt_ids'][1], [1])

    # Frame 3: empty
    assert len(result['gt_ids'][2]) == 0


def test_load_gt_zero_marked(tmp_path):
    gt_file = tmp_path / "gt.txt"
    gt_file.write_text("1,1,10,20,30,40,0,1,0.5\n")
    result = load_mot_gt(str(gt_file), num_timesteps=1)
    np.testing.assert_array_equal(result['gt_extras'][0]['zero_marked'], [0])


def test_load_tracker(tmp_path):
    tracker_file = tmp_path / "seq.txt"
    tracker_file.write_text(
        "1,1,100,200,50,60,0.95,-1,-1,-1\n"
        "2,1,105,205,50,60,0.90,-1,-1,-1\n"
        "2,2,300,400,70,80,0.85,-1,-1,-1\n"
    )
    result = load_mot_tracker(str(tracker_file), num_timesteps=3)

    # Frame 1
    np.testing.assert_array_equal(result['tracker_ids'][0], [1])
    np.testing.assert_allclose(result['tracker_confidences'][0], [0.95], atol=1e-4)

    # Frame 2
    np.testing.assert_array_equal(result['tracker_ids'][1], [1, 2])

    # Frame 3: empty
    assert len(result['tracker_ids'][2]) == 0


def test_load_empty_file(tmp_path):
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("")
    result = load_mot_gt(str(empty_file), num_timesteps=2)
    assert len(result['gt_ids'][0]) == 0
    assert len(result['gt_ids'][1]) == 0
