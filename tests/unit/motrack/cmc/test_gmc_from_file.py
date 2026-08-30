"""
Unit tests for the file-backed GMC algorithm.
"""
import os
import tempfile
import unittest

import numpy as np

from motrack.cmc.algorithms.base import CMCContext
from motrack.cmc.algorithms.gmc_from_file import GMCFromFile, GmcFromFileConfig
from motrack.cmc.components.warp import is_identity_warp

SCENE = 'MOT17-02-FRCNN'
IMAGE_SIZE = (1920, 1080)

# Line format: frame_id, a00, a01, a02, a10, a11, a12
GMC_LINES = [
    '1\t1.0\t0.0\t10.0\t0.0\t1.0\t-5.0',
    '2\t1.0\t0.0\t20.0\t0.0\t1.0\t-8.0',
    '3\t0.99\t0.02\t4.0\t-0.03\t0.98\t3.0',
]


class GMCFromFileTest(unittest.TestCase):
    """
    Tests for GMCFromFile warp lookup and normalization.
    """

    def setUp(self) -> None:
        # pylint: disable=consider-using-with
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp_dir.cleanup)

        path = os.path.join(self._tmp_dir.name, f'GMC-{SCENE}.txt')
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(GMC_LINES))

        self._cmc = GMCFromFile(GmcFromFileConfig(dirpath=self._tmp_dir.name))

    def _context(self, frame_index: int) -> CMCContext:
        return CMCContext(frame_index=frame_index, scene=SCENE, image_size=IMAGE_SIZE)

    def test_first_frame_returns_identity(self) -> None:
        """
        The warp maps frame t-1 to t, which is undefined for the first frame of a scene.
        """
        self.assertTrue(is_identity_warp(self._cmc.apply(self._context(0))))

    def test_row_selection_uses_previous_frame_line(self) -> None:
        """
        Frame `t` must read the warp stored on line `t - 1`.

        Pins the frame-index contract: the returned warp describes the transition into the
        current frame, not out of it.
        """
        width, height = IMAGE_SIZE

        warp = self._cmc.apply(self._context(1))
        np.testing.assert_allclose(warp[:, 2], [10.0 / width, -5.0 / height], atol=1e-9)

        warp = self._cmc.apply(self._context(2))
        np.testing.assert_allclose(warp[:, 2], [20.0 / width, -8.0 / height], atol=1e-9)

    def test_apply_twice_returns_same_warp(self) -> None:
        """
        Repeated calls for the same frame must be idempotent.

        Regression guard: the lookup array is shared across calls, so normalizing it in
        place would silently corrupt the cache and halve the translation each time.
        """
        first = self._cmc.apply(self._context(1))
        second = self._cmc.apply(self._context(1))
        np.testing.assert_array_equal(first, second)

    def test_linear_block_is_conjugated(self) -> None:
        """
        Off-diagonal terms are scaled by the aspect ratio, not left in pixel form.
        """
        width, height = IMAGE_SIZE
        warp = self._cmc.apply(self._context(3))

        self.assertAlmostEqual(float(warp[0, 0]), 0.99, places=6)
        self.assertAlmostEqual(float(warp[1, 1]), 0.98, places=6)
        self.assertAlmostEqual(float(warp[0, 1]), 0.02 * height / width, places=6)
        self.assertAlmostEqual(float(warp[1, 0]), -0.03 * width / height, places=6)

    def test_requires_image_for_dimensions(self) -> None:
        """
        The file-backed GMC never reads pixels, but the frame is still the only source of
        the image dimensions used to normalize the stored pixel-space warps.
        """
        self.assertTrue(GMCFromFile.requires_image)

    def test_requires_scene_and_image_size(self) -> None:
        """
        Scene name and image size are both mandatory.
        """
        with self.assertRaises(AssertionError):
            self._cmc.apply(CMCContext(frame_index=1, scene=None, image_size=IMAGE_SIZE))
        with self.assertRaises(AssertionError):
            self._cmc.apply(CMCContext(frame_index=1, scene=SCENE, image_size=None))

    def test_dancetrack_filename_special_case(self) -> None:
        """
        DanceTrack scene names map to hyphenated GMC filenames.
        """
        # pylint: disable=protected-access
        self.assertEqual(GMCFromFile._get_gmc_filename('dancetrack0004'), 'GMC-dancetrack-0004.txt')
        self.assertEqual(GMCFromFile._get_gmc_filename('MOT17-02-FRCNN'), 'GMC-MOT17-02-FRCNN.txt')


if __name__ == '__main__':
    unittest.main()
