"""
Pairwise IoU computation for bounding boxes in xywh format.
"""
from copy import deepcopy

import numpy as np


def compute_box_ious(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """
    Computes pairwise IoU between two sets of bounding boxes.

    Args:
        bboxes1: Array of shape (N, 4) in xywh format.
        bboxes2: Array of shape (M, 4) in xywh format.

    Returns:
        IoU matrix of shape (N, M).
    """
    if len(bboxes1) == 0 or len(bboxes2) == 0:
        return np.zeros((len(bboxes1), len(bboxes2)))

    # Convert xywh to x0y0x1y1
    bboxes1 = deepcopy(bboxes1)
    bboxes2 = deepcopy(bboxes2)
    bboxes1[:, 2] = bboxes1[:, 0] + bboxes1[:, 2]
    bboxes1[:, 3] = bboxes1[:, 1] + bboxes1[:, 3]
    bboxes2[:, 2] = bboxes2[:, 0] + bboxes2[:, 2]
    bboxes2[:, 3] = bboxes2[:, 1] + bboxes2[:, 3]

    # Pairwise intersection
    min_ = np.minimum(bboxes1[:, np.newaxis, :], bboxes2[np.newaxis, :, :])
    max_ = np.maximum(bboxes1[:, np.newaxis, :], bboxes2[np.newaxis, :, :])
    intersection = np.maximum(min_[..., 2] - max_[..., 0], 0) * np.maximum(min_[..., 3] - max_[..., 1], 0)

    # Areas
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
    union = area1[:, np.newaxis] + area2[np.newaxis, :] - intersection

    # Handle degenerate boxes
    intersection[area1 <= 0 + np.finfo('float').eps, :] = 0
    intersection[:, area2 <= 0 + np.finfo('float').eps] = 0
    intersection[union <= 0 + np.finfo('float').eps] = 0
    union[union <= 0 + np.finfo('float').eps] = 1

    return intersection / union
