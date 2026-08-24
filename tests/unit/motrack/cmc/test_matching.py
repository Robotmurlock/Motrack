"""
Unit tests for descriptor matching.
"""
import unittest

import numpy as np

from motrack.cmc.components.matching import match_descriptors


class MatchDescriptorsTest(unittest.TestCase):
    """
    Nearest-neighbour matching with the ratio test.
    """

    def test_identical_sets_match_one_to_one(self) -> None:
        """
        Every descriptor is its own nearest neighbour, and distinct enough for the ratio test.
        """
        rng = np.random.default_rng(4)
        desc = rng.normal(size=(12, 32)).astype(np.float32)

        pairs = match_descriptors(desc, desc, norm='l2')

        self.assertEqual(len(pairs), 12)
        np.testing.assert_array_equal(pairs[:, 0], pairs[:, 1])

    def test_recovers_a_known_permutation(self) -> None:
        """
        Shuffling the targets must be reflected in the returned indices.
        """
        rng = np.random.default_rng(5)
        desc_a = rng.normal(size=(10, 32)).astype(np.float32)
        permutation = rng.permutation(10)
        desc_b = desc_a[permutation]

        pairs = match_descriptors(desc_a, desc_b, norm='l2')

        for query, target in pairs:
            self.assertEqual(permutation[target], query)

    def test_ratio_test_rejects_an_ambiguous_match(self) -> None:
        """
        A query with two near-identical candidates is ambiguous, and ambiguity predicts a
        wrong match far better than distance alone does.

        The first query sits between two nearly identical targets; the second has one clear
        winner. Only the second should survive.
        """
        desc_a = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32)
        desc_b = np.array([[1.0, 0.0], [-1.0, 0.0], [10.0, 0.0]], dtype=np.float32)

        pairs = match_descriptors(desc_a, desc_b, norm='l2', ratio_threshold=0.8)

        self.assertEqual(len(pairs), 1)
        np.testing.assert_array_equal(pairs[0], [1, 2])

    def test_stricter_ratio_keeps_fewer_matches(self) -> None:
        """
        The ratio threshold has to actually bite, monotonically.
        """
        rng = np.random.default_rng(6)
        desc_a = rng.normal(size=(60, 16)).astype(np.float32)
        desc_b = desc_a + rng.normal(scale=0.05, size=desc_a.shape).astype(np.float32)

        counts = [len(match_descriptors(desc_a, desc_b, norm='l2', ratio_threshold=r)) for r in [0.5, 0.7, 0.9]]

        self.assertLessEqual(counts[0], counts[1])
        self.assertLessEqual(counts[1], counts[2])

    def test_pairs_are_sorted_by_query_index(self) -> None:
        """
        RANSAC samples into this array, so a stable order keeps its results reproducible.
        """
        rng = np.random.default_rng(7)
        desc_a = rng.normal(size=(40, 16)).astype(np.float32)
        desc_b = rng.normal(size=(40, 16)).astype(np.float32)

        pairs = match_descriptors(desc_a, desc_b, norm='l2')

        np.testing.assert_array_equal(pairs[:, 0], np.sort(pairs[:, 0]))

    def test_selection_of_the_two_nearest_is_already_ordered(self) -> None:
        """
        The ratio test assumes `argpartition` returns its two picks nearest-first.

        No reordering is done after the partition, which is only safe because `kth=1` puts
        position 1 in its final sorted place with everything before it no larger. Selecting
        more than two neighbours would void that, and the failure would be silent: best and
        runner-up swap, and every match is rejected instead of raising.
        """
        rng = np.random.default_rng(11)
        for shape in [(200, 50), (200, 3), (200, 500)]:
            with self.subTest(shape=shape):
                distances = rng.integers(0, 5, shape).astype(np.float32)  # ties on purpose
                nearest = np.argpartition(distances, kth=1, axis=1)[:, :2]
                two_best = np.take_along_axis(distances, nearest, axis=1)

                self.assertTrue(bool(np.all(two_best[:, 0] <= two_best[:, 1])))

    def test_batching_does_not_change_the_result(self) -> None:
        """
        Batching exists to bound memory, so it must be invisible in the output.
        """
        rng = np.random.default_rng(8)
        desc_a = rng.normal(size=(50, 16)).astype(np.float32)
        desc_b = rng.normal(size=(30, 16)).astype(np.float32)

        whole = match_descriptors(desc_a, desc_b, norm='l2', batch_size=1000)
        batched = match_descriptors(desc_a, desc_b, norm='l2', batch_size=7)

        np.testing.assert_array_equal(whole, batched)

    def test_hamming_matching_works_on_packed_descriptors(self) -> None:
        """
        ORB descriptors are packed bits, and must match through the same entry point.
        """
        rng = np.random.default_rng(9)
        desc = rng.integers(0, 256, size=(15, 32), dtype=np.uint8)

        pairs = match_descriptors(desc, desc, norm='hamming')

        np.testing.assert_array_equal(pairs[:, 0], pairs[:, 1])

    def test_degenerate_inputs_return_no_matches(self) -> None:
        """
        Fewer than two targets leaves nothing to compare the best match against, so the ratio
        test cannot be evaluated at all. Empty inputs must not raise.
        """
        desc = np.random.default_rng(10).normal(size=(5, 8)).astype(np.float32)
        cases = {
            'empty_query': (np.zeros((0, 8), np.float32), desc),
            'empty_target': (desc, np.zeros((0, 8), np.float32)),
            'single_target': (desc, desc[:1]),
        }
        for name, (query, target) in cases.items():
            with self.subTest(case=name):
                pairs = match_descriptors(query, target, norm='l2')
                self.assertEqual(pairs.shape, (0, 2))
                self.assertEqual(pairs.dtype, np.int32)


if __name__ == '__main__':
    unittest.main()
