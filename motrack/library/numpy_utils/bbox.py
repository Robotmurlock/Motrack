"""
Set of numpy functions applied on bounding box coordinates.
"""
import numpy as np


def affine_transform(warp: np.ndarray, bbox_coords: np.ndarray) -> np.ndarray:
    """
    Applied affine transform to bbox coordinates in format XYXY!

    Args:
        warp: Warp matrix
        bbox_coords: Bounding box coordinates

    Returns:
        Warped bounding box coordinates
    """
    assert bbox_coords.shape == (4,)
    L = np.kron(np.eye(2, dtype=np.float32), warp[:, :2])
    T = np.concatenate([warp[:, 2], warp[:, 2]])
    t_bbox_coords = L @ bbox_coords + T
    return t_bbox_coords
