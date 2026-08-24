"""
Unit tests for the Kalman residual camera motion compensation.

This algorithm reads no frames, so the rendered scene helpers do not apply. Its inputs are
bounding boxes, which makes the ground truth exact: a known warp is applied to a set of boxes
and the algorithm has to recover it.
"""
import unittest

import numpy as np

from motrack.cmc.algorithms.base import CMCContext
from motrack.cmc.algorithms.kf_residual import KFResidualCMC, KFResidualCMCConfig
from motrack.cmc.components.warp import is_identity_warp
from motrack.cmc.factory import cmc_factory
from motrack.library.cv.bbox import BBox, PredBBox
from tests.unit.motrack.cmc.cmc_contract import CMCContractTestMixin

IMAGE_SIZE = (480, 360)


def boxes(centres, size: float = 0.06, conf: float = 0.9):
    """
    Builds detections of a fixed size at the given normalized centres.
    """
    return [
        PredBBox.create(
            bbox=BBox.from_xyxy(cx - size, cy - size, cx + size, cy + size, clip=True),
            label=0,
            conf=conf
        )
        for cx, cy in centres
    ]


def shifted(centres, dx: float, dy: float):
    """
    The same centres under a pure translation.
    """
    return [(cx + dx, cy + dy) for cx, cy in centres]


def context(predictions, detections) -> CMCContext:
    """
    Builds a context the way the tracker does. No frames: this algorithm needs none.
    """
    return CMCContext(
        frame_index=1,
        scene='synthetic',
        prev_frame=None,
        curr_frame=None,
        image_size=IMAGE_SIZE,
        detections=detections,
        tracklet_bbox_predictions=predictions
    )


CENTRES = [(0.20, 0.30), (0.50, 0.25), (0.75, 0.40), (0.35, 0.65), (0.60, 0.70), (0.85, 0.20)]


class KFResidualContractTest(CMCContractTestMixin, unittest.TestCase):
    """
    The shared contract, which this algorithm satisfies through a different route: it returns
    identity for want of correspondences rather than for want of a previous frame.
    """
    CATALOG_KEY = 'kf-residual'


class KFResidualAccuracyTest(unittest.TestCase):
    """
    Recovery of a known camera translation from prediction/detection residuals.
    """

    def _recover(self, dx: float, dy: float, **overrides) -> np.ndarray:
        predictions = boxes(CENTRES)
        detections = boxes(shifted(CENTRES, dx, dy))
        cmc = KFResidualCMC(KFResidualCMCConfig.model_validate(overrides))
        return cmc.apply(context(predictions, detections))

    def test_recovers_a_known_translation(self) -> None:
        """
        Every object displaced identically is camera motion by definition.
        """
        warp = self._recover(0.02, -0.015)
        np.testing.assert_allclose(warp[:, 2], [0.02, -0.015], atol=1e-4)

    def test_warp_is_already_normalized(self) -> None:
        """
        Boxes are normalized, so no pixel conversion happens anywhere in this algorithm.

        A 0.02 shift must come back as 0.02, not as 0.02 * 480.
        """
        warp = self._recover(0.02, 0.0)
        self.assertLess(abs(float(warp[0, 2])), 1.0)

    def test_static_camera_gives_identity(self) -> None:
        """
        Predictions landing on their detections mean the camera did not move.
        """
        self.assertTrue(is_identity_warp(self._recover(0.0, 0.0), atol=1e-6))

    def test_corners_recover_the_same_translation_as_centres(self) -> None:
        """
        Under pure translation the corners carry no information the centre lacks.
        """
        centre = self._recover(0.02, -0.015, points='center')
        corners = self._recover(0.02, -0.015, points='corners')
        np.testing.assert_allclose(centre, corners, atol=1e-5)

    def test_corners_produce_four_points_per_match(self) -> None:
        """
        Corners exist to raise the correspondence count, which is this algorithm's scarcity.
        """
        cmc = KFResidualCMC(KFResidualCMCConfig.model_validate({'points': 'corners'}))
        # pylint: disable=protected-access
        src, _ = cmc._correspondences(boxes(CENTRES), boxes(CENTRES))
        self.assertEqual(len(src), 4 * len(CENTRES))

    def test_median_survives_a_minority_of_bad_associations(self) -> None:
        """
        The translation model uses a median, so it tolerates wrong matches up to half.
        """
        predictions = boxes(CENTRES)
        moved = shifted(CENTRES, 0.02, 0.0)
        # Two of six objects also move on their own, which the motion model failed to predict.
        moved[0] = (moved[0][0] + 0.15, moved[0][1] + 0.12)
        moved[1] = (moved[1][0] - 0.11, moved[1][1] + 0.09)

        cmc = KFResidualCMC(KFResidualCMCConfig())
        warp = cmc.apply(context(predictions, boxes(moved)))

        np.testing.assert_allclose(warp[:, 2], [0.02, 0.0], atol=2e-3)


class KFResidualDegradationTest(unittest.TestCase):
    """
    The failure modes, which for this algorithm are the interesting part.
    """

    def test_too_few_correspondences_gives_identity(self) -> None:
        """
        Correspondences are capped by the number of tracked objects, so scarcity is normal.
        """
        centres = CENTRES[:1]
        warp = KFResidualCMC(KFResidualCMCConfig.model_validate({'min_correspondences': 3})).apply(
            context(boxes(centres), boxes(shifted(centres, 0.02, 0.0)))
        )
        self.assertTrue(is_identity_warp(warp))

    def test_no_tracklets_gives_identity(self) -> None:
        """
        The first frames of a scene have predictions for nothing.
        """
        warp = KFResidualCMC(KFResidualCMCConfig()).apply(context([], boxes(CENTRES)))
        self.assertTrue(is_identity_warp(warp))

    def test_low_confidence_detections_are_dropped(self) -> None:
        """
        A correspondence is only as good as the box it is built from.
        """
        cmc = KFResidualCMC(KFResidualCMCConfig.model_validate({'detection_threshold': 0.6}))
        # pylint: disable=protected-access
        src, _ = cmc._correspondences(boxes(CENTRES), boxes(CENTRES, conf=0.1))
        self.assertEqual(len(src), 0)

    def test_large_motion_breaks_the_association(self) -> None:
        """
        The documented weakness: the association is uncompensated, so it fails exactly when
        camera motion is large enough to pull predictions off their detections.

        At a shift of 0.3 no prediction overlaps its detection, IoU gating matches nothing, and
        the algorithm reports no motion at all - in the one situation where the motion is real.
        """
        predictions = boxes(CENTRES)
        detections = boxes(shifted(CENTRES, 0.30, 0.0))

        warp = KFResidualCMC(KFResidualCMCConfig()).apply(context(predictions, detections))

        self.assertTrue(is_identity_warp(warp))


class KFResidualConfigTest(unittest.TestCase):
    """
    Wiring that does not need the algorithm to run.
    """

    def test_requires_no_image(self) -> None:
        """
        This is the algorithm's whole advantage: no frame is decoded.
        """
        self.assertFalse(cmc_factory('kf-residual', {}).requires_image)

    def test_association_comes_from_the_tracker_catalog(self) -> None:
        """
        Reusing the tracker's association means no second implementation to keep in step.
        """
        cmc = cmc_factory('kf-residual', {'association': {'name': 'iou', 'params': {'match_threshold': 0.5}}})
        # pylint: disable=protected-access
        self.assertIsNotNone(cmc._association)


if __name__ == '__main__':
    unittest.main()
