"""
Association methods utilities. Functions:
- Hungarian algorithm
"""
from typing import Tuple, List

import numpy as np
import scipy

INF = 999_999


def hungarian(cost_matrix: np.ndarray) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Performs Hungarian algorithm on arbitrary `cost_matrix`.

    Args:
        cost_matrix: Any cost matrix

    Returns:
        - List of matched (tracklet, detection) index pairs
        - List of unmatched tracklets indices
        - List of unmatched detection indices
    """
    n_tracklets, n_detections = cost_matrix.shape

    # Augment cost matrix with fictional values so that the linear_sum_assignment always has some solution
    augmentation = INF * np.ones(shape=(n_tracklets, n_tracklets), dtype=np.float32) - np.eye(n_tracklets, dtype=np.float32)
    augmented_cost_matrix = np.hstack([cost_matrix, augmentation])

    row_indices, col_indices = scipy.optimize.linear_sum_assignment(augmented_cost_matrix)

    # All row indices that have values above are matched with augmented part of the matrix hence they are not matched
    matches = []
    matched_tracklets = set()
    matched_detections = set()
    for r_i, c_i in zip(row_indices, col_indices):
        if c_i >= n_detections:
            continue  # augmented match -> no match
        matches.append((r_i, c_i))
        matched_tracklets.add(r_i)
        matched_detections.add(c_i)

    unmatched_tracklets = list(set(range(n_tracklets)) - matched_tracklets)
    unmatched_detections = list(set(range(n_detections)) - matched_detections)

    return matches, unmatched_tracklets, unmatched_detections


def greedy(cost_matrix: np.ndarray) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Performs Greedy algorithm on arbitrary `cost_matrix`.
    In general case this does not give better solution than the Hungarian algorithm
    but instead performs association faster.

    Args:
        cost_matrix: Any cost matrix

    Returns:
        - List of matched (tracklet, detection) index pairs
        - List of unmatched tracklets indices
        - List of unmatched detection indices
    """
    n_tracklets, n_detections = cost_matrix.shape
    matches = []
    matched_tracklets = set()
    matched_detections = set()

    for t_i in range(n_tracklets):
        best_match, best_score = None, None
        for d_i in range(n_detections):
            score = cost_matrix[t_i, d_i]
            if score == np.inf or d_i in matched_detections:
                continue

            if best_score is None or score < best_score:
                best_match = d_i
                best_score = score

        if best_match is not None:
            matches.append((t_i, best_match))
            matched_tracklets.add(t_i)
            matched_detections.add(best_match)

    unmatched_tracklets = list(set(range(n_tracklets)) - matched_tracklets)
    unmatched_detections = list(set(range(n_detections)) - matched_detections)
    return matches, unmatched_tracklets, unmatched_detections
