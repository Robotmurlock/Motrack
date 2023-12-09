import os
from typing import Optional, Dict

import numpy as np

from motrack.cmc.algorithms.base import CameraMotionCompensation
from motrack.cmc.catalog import CMC_CATALOG


@CMC_CATALOG.register('gmc-from-file')
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
        filename = f'GMC-{scene}.txt'

        # DanceTrack naming (special case)
        if scene.startswith('dancetrack'):
            filename = filename.replace('dancetrack', 'dancetrack-')

        return filename

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
            for line_i, line in enumerate(lines):
                tokens = line.strip().split(GMCFromFile.LINE_SEP)[1:]
                for token_i, token in enumerate(tokens):
                    r, c = token_i // 3, token_i % 3
                    warps[line_i, r, c] = float(tokens[token_i])

            gmc_lookup[file] = warps

        return gmc_lookup


    def apply(self, frame: np.ndarray, frame_index: int, scene: Optional[str] = None) -> np.ndarray:
        assert scene is not None, f'Scene name is required in order to load GMC warps from a file!'
        scene_file = self._get_gmc_filename(scene)
        warp = self._gmc_lookup[scene_file][frame_index, :, :]
        warp[0, 2] /= frame.shape[1]
        warp[1, 2] /= frame.shape[0]
        return warp
