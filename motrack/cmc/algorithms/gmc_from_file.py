"""
Load GMC results from file - pre-calculated.
"""
import os
from typing import ClassVar, Dict

import numpy as np
from pydantic import BaseModel, ConfigDict

from motrack.cmc.algorithms.base import CameraMotionCompensation, CMCContext
from motrack.cmc.catalog import CMC_CATALOG
from motrack.cmc.components.warp import identity_warp, pixel_warp_to_normalized


@CMC_CATALOG.register_config('gmc-from-file')
class GmcFromFileConfig(BaseModel):
    """
    Config for file-backed GMC.
    """

    model_config = ConfigDict(extra='forbid')

    dirpath: str


@CMC_CATALOG.register('gmc-from-file')
class GMCFromFile(CameraMotionCompensation):
    """
    Loads precalculated GMC warps from a directory for each scene.

    The stored warps are in pixel coordinates, so they are converted to normalized
    coordinates using the frame dimensions.

    Pixels are never read, but the frame is still required because it is the only source
    of the image dimensions needed for that conversion.
    """
    LINE_SEP = '\t'

    requires_image: ClassVar[bool] = True

    def __init__(self, config: GmcFromFileConfig):
        """
        Args:
            config: Config with the path to the directory where precalculated GMC warps are stored.
        """
        self._gmc_lookup = self._parse_gmc_directory(config.dirpath)

    @staticmethod
    def _get_gmc_filename(scene: str) -> str:
        """
        Gets GMC filename based on the scene name.

        Args:
            scene: Scene name

        Returns:
            GMC filename
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
                for token_i in range(len(tokens)):
                    r, c = token_i // 3, token_i % 3
                    warps[line_i, r, c] = float(tokens[token_i])

            gmc_lookup[file] = warps

        return gmc_lookup

    def apply(self, ctx: CMCContext) -> np.ndarray:
        assert ctx.scene is not None, 'Scene name is required in order to load GMC warps from a file!'
        assert ctx.image_size is not None, 'Image size is required in order to normalize GMC warps!'

        if ctx.frame_index <= 0:
            # The warp maps frame `frame_index - 1` to `frame_index`, which is undefined for the first frame.
            return identity_warp()

        scene_file = self._get_gmc_filename(ctx.scene)
        # Warps are stored per frame transition, so frame `t` uses the warp on line `t - 1`.
        # A copy is mandatory: the lookup is shared across calls and normalization must not mutate it.
        warp = self._gmc_lookup[scene_file][ctx.frame_index - 1, :, :].copy()

        width, height = ctx.image_size
        return pixel_warp_to_normalized(warp, width=width, height=height)
