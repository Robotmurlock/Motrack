"""
Unit tests for descriptor distance metrics.
"""
import unittest

import numpy as np

from motrack.cmc.components.distances import DESCRIPTOR_DISTANCE_CATALOG, descriptor_distances, get_descriptor_distance


def _reference_hamming(desc_a: np.ndarray, desc_b: np.ndarray) -> np.ndarray:
    """
    Straightforward popcount over an XOR, used to check the bit-matmul shortcut.

    This is the implementation the fast path replaces: correct, obvious, and far too slow and
    memory hungry to run per frame.
    """
    distances = np.zeros((len(desc_a), len(desc_b)), dtype=np.float64)
    for i, a in enumerate(desc_a):
        for j, b in enumerate(desc_b):
            distances[i, j] = np.unpackbits(np.bitwise_xor(a, b)).sum()
    return distances


class DescriptorDistanceTest(unittest.TestCase):
    """
    Distance matrices for both descriptor types.
    """

    def test_l2_matches_a_brute_force_loop(self) -> None:
        """
        The expanded form must agree with an explicit pairwise computation.
        """
        rng = np.random.default_rng(0)
        desc_a = rng.normal(size=(7, 16)).astype(np.float32)
        desc_b = rng.normal(size=(5, 16)).astype(np.float32)

        expected = np.array([[np.linalg.norm(a - b) for b in desc_b] for a in desc_a])
        np.testing.assert_allclose(descriptor_distances(desc_a, desc_b, 'l2'), expected, atol=1e-4)

    def test_l2_self_distance_is_negligible(self) -> None:
        """
        Pins the accuracy of the expanded form on realistically large descriptors.

        `||a||^2 + ||b||^2 - 2ab` subtracts two near-equal quantities, so it loses precision
        in float32 where a direct `||a - b||` would not. On unnormalised SIFT descriptors,
        whose magnitude is around 1000, the self-distance comes out around 1 rather than 0 -
        a relative error of about 1e-3.

        That is harmless here: the ratio test compares two distances against each other, so a
        common relative error cancels, and a distance of 1 between identical descriptors is
        still far below any real one. It is worth pinning so the tradeoff stays visible.
        """
        rng = np.random.default_rng(1)
        desc = rng.normal(size=(20, 128)).astype(np.float32) * 100.0
        magnitude = float(np.median(np.linalg.norm(desc, axis=1)))

        self_distance = np.diag(descriptor_distances(desc, desc, 'l2'))

        self.assertGreaterEqual(self_distance.min(), 0.0, 'Clamping must prevent negative squared distances')
        self.assertLess(self_distance.max() / magnitude, 1e-2)

    def test_hamming_matches_an_unpackbits_xor_count(self) -> None:
        """
        Guards the bit-matmul shortcut against the obvious implementation.

        Materialising the (Na, Nb, 32) XOR tensor is 128 MB of interpreted byte operations
        per frame at ORB's usual feature count, so the identity
        `popcount(a ^ b) = sum(a) + sum(b) - 2 * (a . b)` is not optional - but it is easy to
        get subtly wrong.
        """
        rng = np.random.default_rng(2)
        desc_a = rng.integers(0, 256, size=(9, 32), dtype=np.uint8)
        desc_b = rng.integers(0, 256, size=(6, 32), dtype=np.uint8)

        np.testing.assert_allclose(
            descriptor_distances(desc_a, desc_b, 'hamming'),
            _reference_hamming(desc_a, desc_b),
            atol=1e-4
        )

    def test_hamming_is_bounded_by_the_bit_count(self) -> None:
        """
        A 32 byte descriptor cannot differ in more than 256 bits.
        """
        rng = np.random.default_rng(3)
        desc_a = rng.integers(0, 256, size=(10, 32), dtype=np.uint8)
        desc_b = rng.integers(0, 256, size=(10, 32), dtype=np.uint8)

        distances = descriptor_distances(desc_a, desc_b, 'hamming')
        self.assertGreaterEqual(distances.min(), 0.0)
        self.assertLessEqual(distances.max(), 256.0)

    def test_rejects_unknown_norm(self) -> None:
        """
        The norm comes from the detector, so an unknown one is a wiring mistake.
        """
        with self.assertRaisesRegex(ValueError, 'Unknown descriptor norm'):
            descriptor_distances(np.zeros((2, 4), np.float32), np.zeros((2, 4), np.float32), 'cosine')

    def test_rejects_mismatched_descriptor_sizes(self) -> None:
        """
        Descriptors from different detectors cannot be compared.
        """
        with self.assertRaises(AssertionError):
            descriptor_distances(np.zeros((2, 4), np.float32), np.zeros((2, 8), np.float32), 'l2')


class DistanceCatalogTest(unittest.TestCase):
    """
    Metric lookup, which is what makes the set of norms extensible.
    """

    def test_both_norms_are_registered(self) -> None:
        """
        A detector's declared norm has to resolve to a metric, or matching cannot run.
        """
        self.assertEqual(sorted(DESCRIPTOR_DISTANCE_CATALOG.keys), ['hamming', 'l2'])

    def test_getter_returns_a_callable_per_norm(self) -> None:
        """
        The getter is the seam an extra norm would be added through.
        """
        for norm in ['l2', 'hamming']:
            with self.subTest(norm=norm):
                self.assertTrue(callable(get_descriptor_distance(norm)))

    def test_getter_rejects_an_unknown_norm(self) -> None:
        """
        Failing at lookup names the offending norm, rather than failing later on shapes.
        """
        with self.assertRaisesRegex(ValueError, 'Unknown descriptor norm'):
            get_descriptor_distance('cosine')


if __name__ == '__main__':
    unittest.main()
