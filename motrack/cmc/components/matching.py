"""
Descriptor matching between two frames. Establishes correspondences by comparing descriptors. 
Supporting module for feature based camera motion compensation.

Steps:
1. Extract descriptors from the previous and current frames.
2. Compute the distance matrix between the descriptors.
3. Apply the Lowe's ratio test to the distance matrix. If the best candidate is clearly better than the second best, keep the match.

Lowe's ratio test: A match is kept only when the best candidate is clearly better than the second best:
```
best1, best2 = k_nearest_neighbors(distance_matrix, k=2)
ratio = best1 / best2
keep = ratio < ratio_threshold
```

Reference: https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
"""
import numpy as np

from motrack.cmc.components.distances import DescriptorNorm, descriptor_distances

# Number of query descriptors whose distances are materialised at once. A full (Na, Nb)
# float32 matrix at 2000x2000 is 16 MB, which is fine, but the Hamming path needs the
# unpacked bits too, so rows are processed in batches.
DEFAULT_BATCH_SIZE = 512


def match_descriptors(
    desc_a: np.ndarray,
    desc_b: np.ndarray,
    norm: DescriptorNorm,
    ratio_threshold: float = 0.9,
    batch_size: int = DEFAULT_BATCH_SIZE
) -> np.ndarray:
    """
    Matches descriptors with a nearest-neighbour search plus Lowe's ratio test.

    Args:
        desc_a: Query descriptors (Na, D), from the previous frame
        desc_b: Target descriptors (Nb, D), from the current frame
        norm: Norm the descriptors are compared under
        ratio_threshold: Keep a match when `best < ratio_threshold * second_best`. Lower is
            stricter. BoT-SORT uses 0.9; Lowe's original SIFT paper suggests 0.8.
        batch_size: How many query descriptors to process at a time

    Returns:
        Index pairs (M, 2) int32, where column 0 indexes `desc_a` and column 1 indexes
        `desc_b`. Sorted by query index, so the ordering is deterministic - RANSAC samples
        into this array, and a stable order keeps its results reproducible.
    """
    assert 0.0 < ratio_threshold <= 1.0, f'Ratio threshold must be in (0, 1] but got {ratio_threshold}!'

    if len(desc_a) < 2 or len(desc_b) < 2:
        # Fewer than two targets leaves nothing to compare the best match against
        return np.zeros((0, 2), dtype=np.int32)

    pairs = []
    for start in range(0, len(desc_a), batch_size):
        batch = desc_a[start:start + batch_size]
        distances = descriptor_distances(batch, desc_b, norm)

        # Partitioning by the 2nd best is faster than sorting all elements to get the top2
        nearest = np.argpartition(distances, kth=1, axis=1)[:, :2]  
        two_best = np.take_along_axis(distances, nearest, axis=1)

        best_index = nearest[:, 0]
        best, second_best = two_best[:, 0], two_best[:, 1]

        # Lowe's ratio test (d1 / d2 < ratio_threshold)
        accepted = best < ratio_threshold * second_best  #  Avoid division by zero
        query_indices = np.flatnonzero(accepted) + start
        pairs.append(np.stack([query_indices, best_index[accepted]], axis=1))

    if len(pairs) == 0:
        return np.zeros((0, 2), dtype=np.int32)

    return np.concatenate(pairs).astype(np.int32)
