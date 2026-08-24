"""
Tests for the correspondence based camera motion compensation algorithms.

Every algorithm is held to the shared contract from `cmc_contract.py`, and then to accuracy
on a rendered scene. Splitting them this way keeps a construction failure in one algorithm
from hiding the state of the others.
"""
import unittest

import numpy as np

from motrack.cmc.algorithms.feature_matching import FeatureMatchingCMC, FeatureMatchingCMCConfig
from motrack.cmc.algorithms.pylk import PyLKCMC, PyLKCMCConfig
from motrack.cmc.components.warp import is_identity_warp
from motrack.cmc.factory import cmc_factory
from tests.unit.motrack.cmc.cmc_contract import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    CMCContractTestMixin,
    context,
    object_detections,
    render_rigid_scene,
    render_scene,
    static_offsets
)


class IdentityContractTest(CMCContractTestMixin, unittest.TestCase):
    """
    The no-op control still has to honour the contract.
    """
    CATALOG_KEY = 'identity'


class PyLKContractTest(CMCContractTestMixin, unittest.TestCase):
    """
    Optical-flow based compensation.
    """
    CATALOG_KEY = 'pylk'


class FeatureMatchingContractTest(CMCContractTestMixin, unittest.TestCase):
    """
    Descriptor-matching based compensation.
    """
    CATALOG_KEY = 'feature-matching'


class FeatureMatchingOrbContractTest(CMCContractTestMixin, unittest.TestCase):
    """
    The same, with the Hamming descriptor path rather than the L2 one.
    """
    CATALOG_KEY = 'feature-matching'

    def build_cmc(self):
        return cmc_factory(self.CATALOG_KEY, {'feature_detector': {'type': 'orb'}})


class CameraMotionAccuracyTest(unittest.TestCase):
    """
    Recovery of a known rigid scene displacement, for every algorithm and detector.

    The whole scene moves together, which is what camera motion looks like.
    """

    ALGORITHMS = {
        'pylk/shi-tomasi': lambda: PyLKCMC(PyLKCMCConfig()),
        'pylk/orb': lambda: PyLKCMC(PyLKCMCConfig.model_validate({'feature_detector': {'type': 'orb'}})),
        'feature-matching/orb': lambda: FeatureMatchingCMC(FeatureMatchingCMCConfig()),
        'feature-matching/sift': lambda: FeatureMatchingCMC(
            FeatureMatchingCMCConfig.model_validate({'feature_detector': {'type': 'sift'}})
        ),
    }

    def _recover(self, build, shift_px: tuple, detections=None) -> np.ndarray:
        prev = render_rigid_scene((0.0, 0.0))
        curr = render_rigid_scene(shift_px)
        return build().apply(context(frame_index=1, prev_frame=prev, curr_frame=curr, detections=detections))

    def test_warp_is_in_normalized_coordinates(self) -> None:
        """
        Bounding boxes and motion filter states are normalized, so the warp must be too.

        Cropping the scene forward by 8 px moves its content backwards by 8 px, which is a
        translation of -8/480 - not of -8. Returning a pixel space warp would scale every
        translation by the frame dimensions and wreck each tracklet it touched.
        """
        dx, dy = 8.0, -6.0
        expected = [-dx / IMAGE_WIDTH, -dy / IMAGE_HEIGHT]

        for name, build in self.ALGORITHMS.items():
            with self.subTest(algorithm=name):
                warp = self._recover(build, (dx, dy))
                np.testing.assert_allclose(warp[:, 2], expected, atol=3e-3)
                np.testing.assert_allclose(warp[:, :2], np.eye(2), atol=2e-2)

    def test_recovers_small_displacement(self) -> None:
        """
        A displacement well inside every algorithm's working range.
        """
        dx, dy = 3.0, 2.0
        expected = [-dx / IMAGE_WIDTH, -dy / IMAGE_HEIGHT]

        for name, build in self.ALGORITHMS.items():
            with self.subTest(algorithm=name):
                warp = self._recover(build, (dx, dy))
                np.testing.assert_allclose(warp[:, 2], expected, atol=3e-3)

    def test_static_scene_gives_identity(self) -> None:
        """
        Two identical frames must produce no compensation.
        """
        for name, build in self.ALGORITHMS.items():
            with self.subTest(algorithm=name):
                self.assertTrue(is_identity_warp(self._recover(build, (0.0, 0.0)), atol=2e-3))


class ExclusionTest(unittest.TestCase):
    """
    Excluding correspondences that land on detected objects.

    This is the core claim of the study: correspondences on independently moving objects do
    not describe camera motion, and a coherent group of them can outvote the background.
    """

    ALGORITHMS = {
        'pylk': lambda enabled: PyLKCMC(PyLKCMCConfig.model_validate({'exclusion': {'enabled': enabled}})),
        'feature-matching': lambda enabled: FeatureMatchingCMC(
            FeatureMatchingCMCConfig.model_validate({'exclusion': {'enabled': enabled}})
        ),
    }

    def test_moving_objects_are_not_mistaken_for_camera_motion(self) -> None:
        """
        Background still, objects moving: the recovered camera motion must be zero.
        """
        moving = np.tile(np.array([7.0, 0.0], dtype=np.float32), (3, 1))
        prev = render_scene(static_offsets())
        curr = render_scene(moving)

        for name, build in self.ALGORITHMS.items():
            with self.subTest(algorithm=name):
                warp = build(True).apply(
                    context(frame_index=1, prev_frame=prev, curr_frame=curr, detections=object_detections())
                )
                self.assertTrue(is_identity_warp(warp, atol=3e-3))

    def test_detections_are_read_from_the_context(self) -> None:
        """
        Detections change every frame, so they arrive through the context rather than config.

        Passing none must leave the algorithm working, just without exclusion.
        """
        prev = render_rigid_scene((0.0, 0.0))
        curr = render_rigid_scene((4.0, 0.0))

        for name, build in self.ALGORITHMS.items():
            with self.subTest(algorithm=name):
                warp = build(True).apply(context(frame_index=1, prev_frame=prev, curr_frame=curr, detections=None))
                np.testing.assert_allclose(warp[:, 2], [-4.0 / IMAGE_WIDTH, 0.0], atol=3e-3)


class RansacToggleTest(unittest.TestCase):
    """
    RANSAC as a switchable stage, so its contribution can be measured rather than assumed.
    """

    def test_disabling_ransac_is_worse_on_outlier_heavy_input(self) -> None:
        """
        A plain least-squares fit has no way to ignore correspondences on moving objects.

        With the background still and every object moving together, the objects form a
        coherent minority. RANSAC should report no camera motion; the unrobust fit should be
        dragged toward the objects.
        """
        moving = np.tile(np.array([12.0, 0.0], dtype=np.float32), (3, 1))
        prev = render_scene(static_offsets())
        curr = render_scene(moving)
        ctx = context(frame_index=1, prev_frame=prev, curr_frame=curr)

        robust = PyLKCMC(PyLKCMCConfig.model_validate({'ransac': {'enabled': True}})).apply(ctx)
        unrobust = PyLKCMC(PyLKCMCConfig.model_validate({'ransac': {'enabled': False}})).apply(ctx)

        robust_error = abs(float(robust[0, 2]))
        unrobust_error = abs(float(unrobust[0, 2]))

        self.assertLess(robust_error, unrobust_error)
        self.assertTrue(is_identity_warp(robust, atol=3e-3))


class SpatialFilterToggleTest(unittest.TestCase):
    """
    The BoT-SORT geometric heuristic as a switchable stage.
    """

    def test_enabling_it_does_not_break_a_clean_scene(self) -> None:
        """
        On coherent camera motion the filter should remove nothing of consequence.
        """
        prev = render_rigid_scene((0.0, 0.0))
        curr = render_rigid_scene((5.0, 3.0))
        ctx = context(frame_index=1, prev_frame=prev, curr_frame=curr)

        config = FeatureMatchingCMCConfig.model_validate({
            'feature_detector': {'type': 'sift'},
            'spatial_filter': {'enabled': True, 'max_relative': 0.25, 'n_std': 2.5}
        })
        warp = FeatureMatchingCMC(config).apply(ctx)

        np.testing.assert_allclose(warp[:, 2], [-5.0 / IMAGE_WIDTH, -3.0 / IMAGE_HEIGHT], atol=3e-3)


class FeatureMatchingConfigTest(unittest.TestCase):
    """
    Configuration rules that do not need the algorithm to run.
    """

    def test_rejects_a_detector_without_descriptors(self) -> None:
        """
        Matching needs descriptors, and Shi-Tomasi produces only corner locations.

        The error names both the detector and the alternative, because the fix is either to
        change the detector or to switch to the tracking based algorithm.
        """
        with self.assertRaisesRegex(ValueError, 'produces no descriptors'):
            cmc_factory('feature-matching', {'feature_detector': {'type': 'shi-tomasi'}})

    def test_accepts_both_descriptor_detectors(self) -> None:
        """
        SIFT and ORB differ in descriptor norm, and both must be usable.
        """
        for detector, norm in [('sift', 'l2'), ('orb', 'hamming')]:
            with self.subTest(detector=detector):
                cmc = cmc_factory('feature-matching', {'feature_detector': {'type': detector}})
                # pylint: disable=protected-access
                self.assertEqual(cmc._feature_detector.descriptor_norm, norm)


if __name__ == '__main__':
    unittest.main()
