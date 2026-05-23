"""Tests for motrack.eval.similarity."""
import numpy as np
import pytest

from motrack.eval.similarity import compute_box_ious


def test_perfect_overlap():
    boxes = np.array([[10, 20, 30, 40]], dtype=np.float32)
    ious = compute_box_ious(boxes, boxes)
    np.testing.assert_allclose(ious, [[1.0]])


def test_no_overlap():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[20, 20, 10, 10]], dtype=np.float32)
    ious = compute_box_ious(a, b)
    np.testing.assert_allclose(ious, [[0.0]])


def test_partial_overlap():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[5, 0, 10, 10]], dtype=np.float32)
    # Intersection: x=[5,10], y=[0,10] -> 5*10=50; union: 100+100-50=150
    ious = compute_box_ious(a, b)
    np.testing.assert_allclose(ious, [[50.0 / 150.0]], atol=1e-6)


def test_multiple_boxes():
    a = np.array([[0, 0, 10, 10], [20, 20, 10, 10]], dtype=np.float32)
    b = np.array([[0, 0, 10, 10]], dtype=np.float32)
    ious = compute_box_ious(a, b)
    assert ious.shape == (2, 1)
    np.testing.assert_allclose(ious[0, 0], 1.0)
    np.testing.assert_allclose(ious[1, 0], 0.0)


def test_empty_first():
    a = np.empty((0, 4), dtype=np.float32)
    b = np.array([[0, 0, 10, 10]], dtype=np.float32)
    ious = compute_box_ious(a, b)
    assert ious.shape == (0, 1)


def test_empty_second():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.empty((0, 4), dtype=np.float32)
    ious = compute_box_ious(a, b)
    assert ious.shape == (1, 0)


def test_both_empty():
    a = np.empty((0, 4), dtype=np.float32)
    b = np.empty((0, 4), dtype=np.float32)
    ious = compute_box_ious(a, b)
    assert ious.shape == (0, 0)
