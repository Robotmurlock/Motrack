"""
Unit tests for the geometric correspondence filter.
"""
import unittest

import numpy as np

from motrack.cmc.components.spatial_filter import filter_by_displacement

IMAGE_SIZE = (480, 360)


def _correspondences(displacements: np.ndarray, origin: float = 100.0) -> tuple:
    src = np.full((len(displacements), 2), origin, dtype=np.float32)
    return src, src + displacements.astype(np.float32)


class AbsoluteCapTest(unittest.TestCase):
    """
    The frame-relative bound. It does not depend on the other correspondences, so it cannot
    interact with whatever else the pipeline is measuring.
    """

    def test_rejects_displacement_beyond_the_cap(self) -> None:
        """
        A camera cannot move a quarter of the frame between consecutive frames.
        """
        width, height = IMAGE_SIZE
        src, dst = _correspondences(np.array([
            [5.0, 5.0],
            [0.30 * width, 0.0],
            [0.0, 0.30 * height],
        ]))

        keep = filter_by_displacement(src, dst, IMAGE_SIZE, max_relative=0.25)

        np.testing.assert_array_equal(keep, [True, False, False])

    def test_cap_is_per_axis(self) -> None:
        """
        The frame is not square, so the bound differs between the axes.
        """
        width, height = IMAGE_SIZE
        # 0.26 * height is inside the width bound but outside the height bound.
        src, dst = _correspondences(np.array([[0.26 * height, 0.26 * height]]))

        keep = filter_by_displacement(src, dst, IMAGE_SIZE, max_relative=0.25)

        self.assertFalse(keep[0])

    def test_disabling_the_cap_keeps_everything(self) -> None:
        """
        Both stages must be independently switchable.
        """
        src, dst = _correspondences(np.array([[400.0, 300.0], [1.0, 1.0]]))

        keep = filter_by_displacement(src, dst, IMAGE_SIZE, max_relative=None)

        np.testing.assert_array_equal(keep, [True, True])


class StatisticalPassTest(unittest.TestCase):
    """
    The mean/standard-deviation filter, and the reason it is off by default.
    """

    def test_rejects_a_lone_deviating_displacement(self) -> None:
        """
        With a clear majority agreeing, the odd one out is removed.
        """
        displacements = np.array([[4.0, 0.0]] * 20 + [[4.0, 30.0]])
        src, dst = _correspondences(displacements)

        keep = filter_by_displacement(src, dst, IMAGE_SIZE, max_relative=None, n_std=2.5)

        self.assertTrue(keep[:20].all())
        self.assertFalse(keep[20])

    def test_is_symmetric(self) -> None:
        """
        BoT-SORT compares without an absolute value, so it clips only displacements above the
        mean and lets equally wrong ones below it through. That asymmetry looks unintended,
        so both tails are rejected here.
        """
        displacements = np.array([[0.0, 0.0]] * 20 + [[30.0, 0.0], [-30.0, 0.0]])
        src, dst = _correspondences(displacements, origin=200.0)

        keep = filter_by_displacement(src, dst, IMAGE_SIZE, max_relative=None, n_std=2.0)

        self.assertFalse(keep[20], 'Displacement far above the mean must be rejected')
        self.assertFalse(keep[21], 'Displacement far below the mean must be rejected too')

    def test_a_majority_of_outliers_defeats_it(self) -> None:
        """
        Documents the failure mode that keeps this stage off by default.

        Mean and standard deviation are dragged by the very outliers they are meant to catch.
        Once moving objects are the majority, the mean follows them and the filter starts
        rejecting the background instead. RANSAC does not have this failure mode, because a
        consensus over minimal samples does not average anything.
        """
        background = np.array([[2.0, 0.0]] * 5
        )
        objects = np.array([[40.0, 0.0]] * 25)
        src, dst = _correspondences(np.concatenate([background, objects]), origin=200.0)

        keep = filter_by_displacement(src, dst, IMAGE_SIZE, max_relative=None, n_std=1.0)

        self.assertFalse(keep[:5].all(), 'The true background is rejected once outliers dominate')
        self.assertTrue(keep[5:].any(), 'The outlier majority survives')

    def test_identical_displacements_are_all_kept(self) -> None:
        """
        Zero deviation must not divide the whole set away.
        """
        src, dst = _correspondences(np.array([[3.0, -2.0]] * 10))

        keep = filter_by_displacement(src, dst, IMAGE_SIZE, max_relative=None, n_std=2.5)

        self.assertTrue(keep.all())

    def test_disabled_by_default(self) -> None:
        """
        Passing no `n_std` leaves only the absolute cap in play.
        """
        displacements = np.array([[4.0, 0.0]] * 20 + [[4.0, 30.0]])
        src, dst = _correspondences(displacements)

        keep = filter_by_displacement(src, dst, IMAGE_SIZE)

        self.assertTrue(keep.all())


class DegenerateInputTest(unittest.TestCase):
    """
    Inputs that must not raise.
    """

    def test_empty_input(self) -> None:
        """
        No correspondences is a normal outcome, not an error.
        """
        empty = np.zeros((0, 2), dtype=np.float32)

        keep = filter_by_displacement(empty, empty, IMAGE_SIZE, n_std=2.5)

        self.assertEqual(keep.shape, (0,))

    def test_everything_rejected_by_the_cap(self) -> None:
        """
        The statistical pass must cope with an empty surviving set.
        """
        src, dst = _correspondences(np.array([[400.0, 0.0], [420.0, 0.0]]))

        keep = filter_by_displacement(src, dst, IMAGE_SIZE, max_relative=0.25, n_std=2.5)

        self.assertFalse(keep.any())

    def test_mismatched_shapes_are_rejected(self) -> None:
        """
        Source and target must correspond one to one.
        """
        with self.assertRaises(AssertionError):
            filter_by_displacement(np.zeros((3, 2), np.float32), np.zeros((4, 2), np.float32), IMAGE_SIZE)


if __name__ == '__main__':
    unittest.main()
