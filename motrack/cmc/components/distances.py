"""
Distance metrics between descriptors. Supporting module for descriptor matching.

Each metric is registered under the norm it implements, so which norms exist is answered by
the catalog rather than by a chain of conditionals. A detector declares the norm its
descriptors are compared under, and that string is what selects the metric here.

Both implementations avoid materialising an (Na, Nb, D) intermediate: at realistic feature
counts that tensor dominates both the memory and the runtime of the whole matching step.
"""
from typing import Callable, Literal

import numpy as np

from motrack.utils.patterns import DynamicCatalog

DescriptorNorm = Literal['l2', 'hamming']
DescriptorDistance = Callable[[np.ndarray, np.ndarray], np.ndarray]

DESCRIPTOR_DISTANCE_CATALOG = DynamicCatalog()


@DESCRIPTOR_DISTANCE_CATALOG.register('l2')
def _l2_distances(desc_a: np.ndarray, desc_b: np.ndarray) -> np.ndarray:
    """
    Euclidean distances via the expansion ||a - b||^2 = ||a||^2 + ||b||^2 - 2ab.

    One matmul, no (Na, Nb, D) temporary.

    Args:
        desc_a: Query descriptors (Na, D)
        desc_b: Target descriptors (Nb, D)

    Returns:
        Distance matrix (Na, Nb)
    """
    a = desc_a.astype(np.float32, copy=False)
    b = desc_b.astype(np.float32, copy=False)

    squared = np.sum(a * a, axis=1)[:, None] + np.sum(b * b, axis=1)[None, :] - 2.0 * (a @ b.T)
    return np.sqrt(np.maximum(squared, 0.0))


@DESCRIPTOR_DISTANCE_CATALOG.register('hamming')
def _hamming_distances(desc_a: np.ndarray, desc_b: np.ndarray) -> np.ndarray:
    """
    Hamming distances between packed binary descriptors.

    The obvious implementation, `popcount(a ^ b)` over an (Na, Nb, D) tensor, is unusable
    here: for 2000 ORB descriptors of 32 bytes that is 128 MB of interpreted byte operations
    per frame. Unpacking to bits once turns the popcount into a single matmul, since for
    binary vectors `popcount(a ^ b) = sum(a) + sum(b) - 2 * (a . b)`.

    Args:
        desc_a: Query descriptors (Na, D) uint8
        desc_b: Target descriptors (Nb, D) uint8

    Returns:
        Distance matrix (Na, Nb)
    """
    bits_a = np.unpackbits(desc_a, axis=1).astype(np.float32)
    bits_b = np.unpackbits(desc_b, axis=1).astype(np.float32)

    ones_a = np.sum(bits_a, axis=1)[:, None]
    ones_b = np.sum(bits_b, axis=1)[None, :]
    return ones_a + ones_b - 2.0 * (bits_a @ bits_b.T)


def get_descriptor_distance(norm: DescriptorNorm) -> DescriptorDistance:
    """
    Looks up the distance metric registered under a norm.

    Args:
        norm: Which norm the descriptors are compared under

    Returns:
        Distance function taking two descriptor sets and returning their distance matrix

    Raises:
        ValueError: If the norm is unknown.
    """
    if norm not in DESCRIPTOR_DISTANCE_CATALOG.keys:
        raise ValueError(
            f'Unknown descriptor norm "{norm}". Expected one of: {", ".join(DESCRIPTOR_DISTANCE_CATALOG.keys)}.'
        )

    return DESCRIPTOR_DISTANCE_CATALOG[norm]


def descriptor_distances(desc_a: np.ndarray, desc_b: np.ndarray, norm: DescriptorNorm) -> np.ndarray:
    """
    Computes the full distance matrix between two descriptor sets.

    Args:
        desc_a: Query descriptors (Na, D)
        desc_b: Target descriptors (Nb, D)
        norm: Which norm the descriptors are compared under. SIFT descriptors are float and
            use 'l2'; ORB descriptors are packed bits and use 'hamming'. The detector
            declares which one applies.

    Returns:
        Distance matrix (Na, Nb)

    Raises:
        ValueError: If the norm is unknown.
    """
    assert desc_a.ndim == 2 and desc_b.ndim == 2, f'Expected 2D descriptors but got {desc_a.shape} and {desc_b.shape}!'
    assert desc_a.shape[1] == desc_b.shape[1], f'Descriptor sizes differ: {desc_a.shape[1]} vs {desc_b.shape[1]}!'

    return get_descriptor_distance(norm)(desc_a, desc_b)
