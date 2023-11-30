"""
Numpy IO functions.
"""
from pathlib import Path

import numpy as np


def load_npy(path: str) -> np.ndarray:
    """
    Loads numpy array from given path.

    Args:
        path: Path

    Returns:
        Loaded numpy array.
    """
    with open(path, 'rb') as f:
        return np.load(f)


def store_npy(data: np.ndarray, path: str) -> None:
    """
    Stores numpy array in path.

    Args:
        data: Data
        path: Path
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        np.save(f, data)
