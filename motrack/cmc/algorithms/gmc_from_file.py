import os
from typing import Optional, Dict

import numpy as np

from motrack.cmc.algorithms.base import CameraMotionCompensation


class GMCFromFile(CameraMotionCompensation):
    """
    Loads precalculated GMC warps from a directory for each scene.
    """
    LINE_SEP = '\t'

    def __init__(self, dirpath: str):
        """
        Args:
            dirpath: Path to directory where precalculated GMC warps are stored.
        """
        self._gmc_lookup = self._parse_gmc_directory(dirpath)

    @staticmethod
    def _get_gmc_filename(scene: str) -> str:
        """
        Gets GMC filename based on the scene name.

        Args:
            scene: Scene name

        Returns:

        """
        return f'GMC-{scene}.txt'

    @staticmethod
    def _parse_gmc_directory(path: str) -> Dict[str, np.ndarray]:
        """
        Parses all files in GMC directory.

        Args:
            path: Path to GMC files.

        Returns:
            GMC warp lookup
        """
        gmc_lookup = {}

        files = os.listdir(path)
        for file in files:
            filepath = os.path.join(path, file)
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = [line for line in f.readlines() if len(line) > 0]

            n_lines = len(lines)
            warps = np.zeros(shape=(n_lines, 2, 3), dtype=np.float32)
            for line in lines:
                tokens = line.split(GMCFromFile.LINE_SEP)[1:]
                for token_i, token in enumerate(tokens):
                    r, c = token_i // 3, token_i % 3
                    warps[r, c] = float(tokens)

            gmc_lookup[file] = warps

        return gmc_lookup


    def apply(self, frame: np.ndarray, frame_index: int, scene: Optional[str] = None) -> np.ndarray:
        _ = frame  # Ignored
        assert scene is not None, f'Scene name is required in order to load GMC warps from a file!'
        return self._gmc_lookup[scene][frame_index, :, :]
