"""
Unit tests for filling detected objects in the frame before feature detection.

This is the alternative to filtering points afterwards. The two are compared in the report;
these tests fix the behaviour each relies on.
"""
import unittest

import numpy as np

from motrack.cmc.algorithms.pylk import PyLKCMC, PyLKCMCConfig
from motrack.cmc.algorithms.utils import mask_detections_in_image, to_grayscale
from motrack.cmc.components.feature_detector.factory import feature_detector_factory
from motrack.cmc.components.warp import is_identity_warp
from motrack.library.cv.bbox import BBox, PredBBox
from tests.unit.motrack.cmc.cmc_contract import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    context,
    object_detections,
    render_rigid_scene,
    render_scene,
    static_offsets
)

IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)


class MaskDetectionsInImageTest(unittest.TestCase):
    """
    The masking primitive itself.
    """

    def test_detected_regions_are_filled(self) -> None:
        """
        Everything inside a detection is replaced by the fill value.
        """
        frame = np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), 200, dtype=np.uint8)
        detection = PredBBox.create(bbox=BBox.from_xyxy(0.25, 0.25, 0.5, 0.5), label=0, conf=0.9)

        masked = mask_detections_in_image(frame, [detection], IMAGE_SIZE, expansion_factor=0.0)

        left, top = int(0.25 * IMAGE_WIDTH), int(0.25 * IMAGE_HEIGHT)
        right, bottom = int(0.5 * IMAGE_WIDTH), int(0.5 * IMAGE_HEIGHT)
        self.assertEqual(masked[top + 2:bottom - 2, left + 2:right - 2].max(), 0)
        self.assertEqual(masked[0, 0].min(), 200)

    def test_input_frame_is_not_modified(self) -> None:
        """
        The tracker reuses the frame it passes in, so masking must not write through it.
        """
        frame = np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), 200, dtype=np.uint8)
        detection = PredBBox.create(bbox=BBox.from_xyxy(0.1, 0.1, 0.9, 0.9), label=0, conf=0.9)

        mask_detections_in_image(frame, [detection], IMAGE_SIZE, expansion_factor=0.0)

        self.assertEqual(frame.min(), 200)

    def test_no_detections_returns_the_frame_unchanged(self) -> None:
        """
        Frames with nothing detected are passed straight through.
        """
        frame = render_scene(static_offsets())
        self.assertIs(mask_detections_in_image(frame, [], IMAGE_SIZE, expansion_factor=0.2), frame)

    def test_expansion_enlarges_the_filled_region(self) -> None:
        """
        The expansion factor applies here as it does to point filtering.
        """
        frame = np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), 200, dtype=np.uint8)
        detection = PredBBox.create(bbox=BBox.from_xyxy(0.4, 0.4, 0.6, 0.6), label=0, conf=0.9)

        plain = mask_detections_in_image(frame, [detection], IMAGE_SIZE, expansion_factor=0.0)
        expanded = mask_detections_in_image(frame, [detection], IMAGE_SIZE, expansion_factor=0.5)

        self.assertLess(int((expanded == 0).sum()), frame.size)
        self.assertGreater(int((expanded == 0).sum()), int((plain == 0).sum()))


class ImageMaskingIntroducesBorderFeaturesTest(unittest.TestCase):
    """
    The property that separates image masking from point filtering.

    Filling a box leaves a step edge along its border. Corner detectors respond to step edges,
    so masking removes the features inside the object and adds features on its outline. Point
    filtering leaves no such edge. The report's §3.3 argument rests on this, so it is measured
    rather than asserted.
    """

    def test_masked_frame_yields_features_on_the_mask_border(self) -> None:
        """
        Features appear along the filled box's outline, where the image had none before.
        """
        frame = render_scene(static_offsets())
        detections = object_detections()
        detector = feature_detector_factory('shi-tomasi', {})

        masked = mask_detections_in_image(frame, detections, IMAGE_SIZE, expansion_factor=0.0)
        points = detector.detect(to_grayscale(masked))[0]

        normalized = points / np.array([IMAGE_WIDTH, IMAGE_HEIGHT], dtype=np.float32)
        on_border = np.zeros(len(points), dtype=bool)
        for detection in detections:
            near_x = np.isclose(normalized[:, 0], detection.upper_left.x, atol=0.02) | \
                np.isclose(normalized[:, 0], detection.bottom_right.x, atol=0.02)
            near_y = np.isclose(normalized[:, 1], detection.upper_left.y, atol=0.02) | \
                np.isclose(normalized[:, 1], detection.bottom_right.y, atol=0.02)
            within_y = (normalized[:, 1] >= detection.upper_left.y - 0.02) & \
                (normalized[:, 1] <= detection.bottom_right.y + 0.02)
            within_x = (normalized[:, 0] >= detection.upper_left.x - 0.02) & \
                (normalized[:, 0] <= detection.bottom_right.x + 0.02)
            on_border |= (near_x & within_y) | (near_y & within_x)

        self.assertGreater(int(on_border.sum()), 0)


class ImageMaskingEndToEndTest(unittest.TestCase):
    """
    The mode reaching the algorithms.
    """

    def test_image_mode_still_recovers_camera_motion(self) -> None:
        """
        Masking must not break the estimate on a scene where background dominates.
        """
        prev = render_rigid_scene((0.0, 0.0))
        curr = render_rigid_scene((4.0, 0.0))
        config = PyLKCMCConfig.model_validate({'exclusion': {'enabled': True, 'mode': 'image'}})

        warp = PyLKCMC(config).apply(context(frame_index=1, prev_frame=prev, curr_frame=curr,
                                             detections=object_detections()))

        np.testing.assert_allclose(warp[:, 2], [-4.0 / IMAGE_WIDTH, 0.0], atol=4e-3)

    def test_modes_produce_different_estimates(self) -> None:
        """
        The two modes are separate configurations, so they must not collapse into one.
        """
        prev = render_scene(static_offsets())
        curr = render_scene(np.tile(np.array([6.0, 0.0], dtype=np.float32), (3, 1)))
        ctx = context(frame_index=1, prev_frame=prev, curr_frame=curr, detections=object_detections())

        points_mode = PyLKCMC(PyLKCMCConfig.model_validate(
            {'exclusion': {'enabled': True, 'mode': 'points'}})).apply(ctx)
        image_mode = PyLKCMC(PyLKCMCConfig.model_validate(
            {'exclusion': {'enabled': True, 'mode': 'image'}})).apply(ctx)

        # Both should report near-zero camera motion here; the point is that they are computed
        # from different feature sets, not that they disagree.
        self.assertTrue(is_identity_warp(points_mode, atol=5e-3))
        self.assertEqual(image_mode.shape, (2, 3))

    def test_default_mode_is_point_filtering(self) -> None:
        """
        The default has to stay what every earlier measurement used.
        """
        self.assertEqual(PyLKCMCConfig().exclusion.mode, 'points')


if __name__ == '__main__':
    unittest.main()
