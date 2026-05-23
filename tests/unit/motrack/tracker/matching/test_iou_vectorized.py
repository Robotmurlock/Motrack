"""
Tests for IoU cost matrix vectorization correctness.

Verifies that the vectorized form_cost_matrix() produces identical results
to the per-element Python implementation for all IoU-based matchers.
"""
import copy
import unittest

import numpy as np

from motrack.library.cv.bbox import BBox, PredBBox
from motrack.tracker.matching.algorithms.iou import (
    IoUAssociation, IoUAssociationConfig,
    HMIoUAssociation, HMIoUAssociationConfig,
    DecayIoU, DecayIoUAssociationConfig,
)
from motrack.tracker.tracklet import Tracklet, TrackletState


def _make_bboxes(n: int, seed: int = 42) -> list:
    rng = np.random.RandomState(seed)
    bboxes = []
    for _ in range(n):
        cx, cy = rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)
        w, h = rng.uniform(0.02, 0.15), rng.uniform(0.02, 0.15)
        x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
        x2, y2 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
        label = int(rng.randint(0, 3))
        conf = float(rng.uniform(0.3, 1.0))
        bboxes.append(PredBBox.create(BBox.from_xyxy(x1, y1, x2, y2), label=label, conf=conf))
    return bboxes


def _make_tracklets(bboxes: list, lost_times: list = None) -> list:
    tracklets = []
    for i, bbox in enumerate(bboxes):
        t = Tracklet(bbox=copy.deepcopy(bbox), frame_index=10, _id=i, state=TrackletState.ACTIVE)
        if lost_times is not None:
            for _ in range(lost_times[i]):
                t.update(copy.deepcopy(bbox), t.frame_index, state=TrackletState.LOST)
        tracklets.append(t)
    return tracklets


def _slow_form_cost_matrix(matcher, tracklet_bboxes, detections, tracklets):
    """Reference: compute cost matrix element-by-element using _calc_score."""
    n_t, n_d = len(tracklet_bboxes), len(detections)
    cost_matrix = np.zeros((n_t, n_d), dtype=np.float32)
    for t_i in range(n_t):
        for d_i in range(n_d):
            if not matcher._can_match_labels(tracklet_bboxes[t_i].label, detections[d_i].label):
                cost_matrix[t_i][d_i] = np.inf
                continue
            cost_matrix[t_i][d_i] = matcher._calc_score(tracklets[t_i], tracklet_bboxes[t_i], detections[d_i])
    return cost_matrix


class TestBatchIoU(unittest.TestCase):

    def test_batch_iou_matches_per_element(self):
        bboxes_a = _make_bboxes(30, seed=42)
        bboxes_b = _make_bboxes(40, seed=123)
        expected = np.array([[a.iou(b) for b in bboxes_b] for a in bboxes_a])
        a_xyxy = np.array([b.as_numpy_xyxy() for b in bboxes_a])
        b_xyxy = np.array([b.as_numpy_xyxy() for b in bboxes_b])
        actual = BBox.batch_iou(a_xyxy, b_xyxy)
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_batch_iou_non_overlapping(self):
        a = np.array([[0.0, 0.0, 0.1, 0.1]], dtype=np.float32)
        b = np.array([[0.5, 0.5, 0.6, 0.6]], dtype=np.float32)
        self.assertEqual(BBox.batch_iou(a, b)[0, 0], 0.0)

    def test_batch_iou_identical(self):
        a = np.array([[0.2, 0.2, 0.4, 0.4]], dtype=np.float32)
        np.testing.assert_allclose(BBox.batch_iou(a, a)[0, 0], 1.0, atol=1e-6)

    def test_batch_iou_empty(self):
        a = np.zeros((0, 4), dtype=np.float32)
        b = np.array([[0.1, 0.1, 0.2, 0.2]], dtype=np.float32)
        self.assertEqual(BBox.batch_iou(a, b).shape, (0, 1))


class TestIoUAssociationVectorized(unittest.TestCase):

    def _assert_cost_matrices_equal(self, matcher, n_t, n_d, seed_t=42, seed_d=123):
        tracklet_bboxes = _make_bboxes(n_t, seed=seed_t)
        detections = _make_bboxes(n_d, seed=seed_d)
        tracklets = _make_tracklets(tracklet_bboxes)
        expected = _slow_form_cost_matrix(matcher, tracklet_bboxes, detections, tracklets)
        actual = matcher.form_cost_matrix(tracklet_bboxes, detections, tracklets=tracklets)
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_iou_small(self):
        self._assert_cost_matrices_equal(IoUAssociation(IoUAssociationConfig(match_threshold=0.3)), 10, 10)

    def test_iou_medium(self):
        self._assert_cost_matrices_equal(IoUAssociation(IoUAssociationConfig(match_threshold=0.3)), 50, 50)

    def test_iou_large(self):
        self._assert_cost_matrices_equal(IoUAssociation(IoUAssociationConfig(match_threshold=0.3)), 100, 100)

    def test_iou_asymmetric(self):
        self._assert_cost_matrices_equal(IoUAssociation(IoUAssociationConfig(match_threshold=0.3)), 20, 80)

    def test_iou_fuse_score(self):
        self._assert_cost_matrices_equal(IoUAssociation(IoUAssociationConfig(match_threshold=0.3, fuse_score=True)), 50, 50)

    def test_iou_empty_tracklets(self):
        matcher = IoUAssociation(IoUAssociationConfig(match_threshold=0.3))
        result = matcher.form_cost_matrix([], _make_bboxes(10), tracklets=[])
        self.assertEqual(result.shape, (0, 10))

    def test_iou_empty_detections(self):
        matcher = IoUAssociation(IoUAssociationConfig(match_threshold=0.3))
        bboxes = _make_bboxes(10)
        result = matcher.form_cost_matrix(bboxes, [], tracklets=_make_tracklets(bboxes))
        self.assertEqual(result.shape, (10, 0))

    def test_iou_label_gating(self):
        cfg = IoUAssociationConfig(match_threshold=0.3, label_gating=[(0, 1)])
        self._assert_cost_matrices_equal(IoUAssociation(cfg), 30, 30)


class TestHMIoUAssociationVectorized(unittest.TestCase):

    def test_hmiou_small(self):
        matcher = HMIoUAssociation(HMIoUAssociationConfig(match_threshold=0.3))
        bboxes_t, bboxes_d = _make_bboxes(10, 42), _make_bboxes(10, 123)
        tracklets = _make_tracklets(bboxes_t)
        expected = _slow_form_cost_matrix(matcher, bboxes_t, bboxes_d, tracklets)
        actual = matcher.form_cost_matrix(bboxes_t, bboxes_d, tracklets=tracklets)
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_hmiou_large(self):
        matcher = HMIoUAssociation(HMIoUAssociationConfig(match_threshold=0.3))
        bboxes_t, bboxes_d = _make_bboxes(100, 42), _make_bboxes(100, 123)
        tracklets = _make_tracklets(bboxes_t)
        expected = _slow_form_cost_matrix(matcher, bboxes_t, bboxes_d, tracklets)
        actual = matcher.form_cost_matrix(bboxes_t, bboxes_d, tracklets=tracklets)
        np.testing.assert_allclose(actual, expected, atol=1e-6)


class TestDecayIoUVectorized(unittest.TestCase):

    def test_decay_iou_no_expansion(self):
        cfg = DecayIoUAssociationConfig(min_threshold=0.2, max_threshold=0.5, threshold_decay=0.05)
        matcher = DecayIoU(cfg)
        bboxes_t, bboxes_d = _make_bboxes(30, 42), _make_bboxes(30, 123)
        tracklets = _make_tracklets(bboxes_t, lost_times=[i % 5 for i in range(30)])
        expected = _slow_form_cost_matrix(matcher, bboxes_t, bboxes_d, tracklets)
        actual = matcher.form_cost_matrix(bboxes_t, bboxes_d, tracklets=tracklets)
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_decay_iou_with_expansion(self):
        cfg = DecayIoUAssociationConfig(min_threshold=0.2, max_threshold=0.5, threshold_decay=0.05, expansion_rate=0.1)
        matcher = DecayIoU(cfg)
        bboxes_t, bboxes_d = _make_bboxes(30, 42), _make_bboxes(30, 123)
        tracklets = _make_tracklets(bboxes_t, lost_times=[i % 5 for i in range(30)])
        expected = _slow_form_cost_matrix(matcher, bboxes_t, bboxes_d, tracklets)
        actual = matcher.form_cost_matrix(bboxes_t, bboxes_d, tracklets=tracklets)
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_decay_iou_with_fuse_score(self):
        cfg = DecayIoUAssociationConfig(min_threshold=0.2, max_threshold=0.5, threshold_decay=0.05, fuse_score=True)
        matcher = DecayIoU(cfg)
        bboxes_t, bboxes_d = _make_bboxes(30, 42), _make_bboxes(30, 123)
        tracklets = _make_tracklets(bboxes_t, lost_times=[i % 5 for i in range(30)])
        expected = _slow_form_cost_matrix(matcher, bboxes_t, bboxes_d, tracklets)
        actual = matcher.form_cost_matrix(bboxes_t, bboxes_d, tracklets=tracklets)
        np.testing.assert_allclose(actual, expected, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
