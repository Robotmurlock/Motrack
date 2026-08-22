"""
Unit tests for CMC integration into the tracker: lifecycle, context contents and the
guarantee that the CMC plumbing is a no-op when no motion is compensated.
"""
import unittest
from typing import List, Optional
from unittest.mock import MagicMock

import numpy as np

from motrack.cmc.algorithms.base import CameraMotionCompensation, CMCContext
from motrack.library.cv.bbox import BBox, PredBBox
from motrack.tracker.trackers.algorithms.sort import SortTracker, SortTrackerConfig
from motrack.tracker.tracklet import Tracklet

IMAGE_SIZE = (1920, 1080)


def _detections(offset: float) -> List[PredBBox]:
    """
    Creates a deterministic pair of detections translated by `offset`.
    """
    return [
        PredBBox.create(BBox.from_xywh(0.10 + offset, 0.20, 0.05, 0.10, clip=False), label=0, conf=0.95),
        PredBBox.create(BBox.from_xywh(0.50 + offset, 0.60, 0.06, 0.12, clip=False), label=0, conf=0.90)
    ]


def _frame() -> np.ndarray:
    """
    Creates a dummy frame. Contents are irrelevant - no test here inspects pixels.
    """
    width, height = IMAGE_SIZE
    return np.zeros(shape=(height, width, 3), dtype=np.uint8)


def _run_scene(tracker: SortTracker, scene: str, n_frames: int = 8) -> List[Tracklet]:
    """
    Runs a tracker over a scripted scene of steadily translating detections.
    """
    tracker.reset_state()
    tracker.set_scene(scene)

    tracklets: List[Tracklet] = []
    frame = _frame()
    for index in range(n_frames):
        tracklets = tracker.track(
            tracklets=tracklets,
            detections=_detections(0.01 * index),
            frame_index=index + 1,
            frame=frame
        )

    return tracklets


def _summarize(tracklets: List[Tracklet]) -> List[tuple]:
    """
    Reduces tracklets to a comparable summary of ids and coordinates.
    """
    return sorted(
        (t.id, t.state, tuple(np.round(t.bbox.as_numpy_xyxy(dtype=np.float64), 12)))
        for t in tracklets
    )


def _config(cmc: Optional[dict]) -> SortTrackerConfig:
    return SortTrackerConfig.model_validate({
        'initialization_threshold': 1,
        'remember_threshold': 10,
        'cmc': cmc
    })


class IdentityCMCEquivalenceTest(unittest.TestCase):
    """
    The identity CMC must be indistinguishable from having no CMC at all.
    """

    def test_identity_cmc_matches_no_cmc_exactly(self) -> None:
        """
        Running with `identity` produces bit-identical tracklets to running without CMC.

        This is the regression test for the whole CMC refactor: interface change, frame
        index contract, lifecycle hook and detection plumbing are all exercised, and the
        identity warp is exact in float32 so equality is exact rather than approximate.
        """
        without_cmc = _summarize(_run_scene(SortTracker(_config(None)), 'scene-a'))
        with_identity = _summarize(_run_scene(SortTracker(_config({'name': 'identity'})), 'scene-a'))

        self.assertEqual(without_cmc, with_identity)

    def test_identity_cmc_matches_no_cmc_across_scenes(self) -> None:
        """
        Same as above, but across two scenes, so `reset_state` participates.
        """
        tracker_plain, tracker_identity = SortTracker(_config(None)), SortTracker(_config({'name': 'identity'}))

        for scene in ['scene-a', 'scene-b']:
            with self.subTest(scene=scene):
                plain = _summarize(_run_scene(tracker_plain, scene))
                identity = _summarize(_run_scene(tracker_identity, scene))
                self.assertEqual(plain, identity)


class TrackerCMCLifecycleTest(unittest.TestCase):
    """
    Tests for the tracker-side CMC lifecycle and context construction.
    """

    def setUp(self) -> None:
        self._tracker = SortTracker(_config({'name': 'identity'}))

    def test_reset_state_clears_filter_states_and_resets_cmc(self) -> None:
        """
        Per-scene state must not leak between scenes.

        Motion filter states used to accumulate for the whole run, so CMC kept warping
        states of long-dead objects and the per-frame cost grew from scene to scene.
        """
        # pylint: disable=protected-access
        _run_scene(self._tracker, 'scene-a')
        self.assertGreater(len(self._tracker._filter_states), 0)

        spy = MagicMock(spec=CameraMotionCompensation)
        spy.requires_image = False
        self._tracker._cmc = spy

        self._tracker.reset_state()

        self.assertEqual(len(self._tracker._filter_states), 0)
        self.assertIsNone(self._tracker._prev_frame)
        self.assertIsNone(self._tracker._prev_frame_index)
        spy.reset.assert_called_once()

    def test_cmc_receives_expected_context(self) -> None:
        """
        The context carries the zero-based current frame index, the scene, the frame, its
        size, the raw detections and one prediction per surviving tracklet.
        """
        # pylint: disable=protected-access
        spy = MagicMock(spec=CameraMotionCompensation)
        spy.requires_image = True
        spy.apply.return_value = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        self._tracker.reset_state()
        self._tracker._cmc = spy
        self._tracker.set_scene('scene-a')

        frame = _frame()
        tracklets: List[Tracklet] = []
        for index in range(3):
            tracklets = self._tracker.track(
                tracklets=tracklets,
                detections=_detections(0.01 * index),
                frame_index=index + 1,
                frame=frame
            )

        contexts = [call.args[0] for call in spy.apply.call_args_list]
        self.assertEqual([ctx.frame_index for ctx in contexts], [0, 1, 2])

        for ctx in contexts:
            self.assertIsInstance(ctx, CMCContext)
            self.assertEqual(ctx.scene, 'scene-a')
            self.assertIs(ctx.curr_frame, frame)
            self.assertEqual(ctx.image_size, IMAGE_SIZE)
            self.assertEqual(len(ctx.detections), 2)

        # No tracklets exist on the first frame, so there is nothing to predict yet.
        self.assertEqual(len(contexts[0].tracklet_bbox_predictions), 0)
        self.assertEqual(len(contexts[-1].tracklet_bbox_predictions), 2)

    def test_previous_frame_is_supplied_by_the_tracker(self) -> None:
        """
        The tracker caches the previous frame so algorithms do not have to.

        It is absent on the first frame of a scene, and present from then on.
        """
        # pylint: disable=protected-access
        spy = MagicMock(spec=CameraMotionCompensation)
        spy.requires_image = True
        spy.apply.return_value = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        self._tracker.reset_state()
        self._tracker._cmc = spy
        self._tracker.set_scene('scene-a')

        frames = [_frame() for _ in range(3)]
        tracklets: List[Tracklet] = []
        for index, frame in enumerate(frames):
            tracklets = self._tracker.track(
                tracklets=tracklets,
                detections=_detections(0.01 * index),
                frame_index=index + 1,
                frame=frame
            )

        contexts = [call.args[0] for call in spy.apply.call_args_list]

        self.assertIsNone(contexts[0].prev_frame, 'Nothing precedes the first frame of a scene')
        self.assertIs(contexts[1].prev_frame, frames[0])
        self.assertIs(contexts[2].prev_frame, frames[1])

    def test_previous_frame_is_dropped_across_a_frame_index_gap(self) -> None:
        """
        Two non-adjacent frames cannot be compared, so the gap is reported as no previous
        frame at all. Centralising this means no algorithm has to check it.
        """
        # pylint: disable=protected-access
        spy = MagicMock(spec=CameraMotionCompensation)
        spy.requires_image = True
        spy.apply.return_value = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        self._tracker.reset_state()
        self._tracker._cmc = spy
        self._tracker.set_scene('scene-a')

        for frame_index in [1, 2, 9]:
            self._tracker.track(tracklets=[], detections=_detections(0.0), frame_index=frame_index, frame=_frame())

        contexts = [call.args[0] for call in spy.apply.call_args_list]

        self.assertIsNone(contexts[0].prev_frame)
        self.assertIsNotNone(contexts[1].prev_frame)
        self.assertIsNone(contexts[2].prev_frame, 'A jump from frame 1 to 8 leaves no adjacent predecessor')

    def test_previous_frame_is_cleared_between_scenes(self) -> None:
        """
        The first frame of a new scene must not be compared against the last frame of the
        previous one.
        """
        # pylint: disable=protected-access
        spy = MagicMock(spec=CameraMotionCompensation)
        spy.requires_image = True
        spy.apply.return_value = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        self._tracker.reset_state()
        self._tracker._cmc = spy

        for scene in ['scene-a', 'scene-b']:
            self._tracker.reset_state()
            self._tracker.set_scene(scene)
            for index in range(2):
                self._tracker.track(tracklets=[], detections=_detections(0.0), frame_index=index + 1, frame=_frame())

        contexts = [call.args[0] for call in spy.apply.call_args_list]
        first_frame_of_each_scene = [contexts[0], contexts[2]]

        for ctx in first_frame_of_each_scene:
            self.assertIsNone(ctx.prev_frame)

    def test_raises_when_frame_required_but_missing(self) -> None:
        """
        A CMC that needs pixels must fail fast instead of crashing on a None frame later.
        """
        # pylint: disable=protected-access
        spy = MagicMock(spec=CameraMotionCompensation)
        spy.requires_image = True
        self._tracker._cmc = spy

        with self.assertRaisesRegex(ValueError, 'requires video frames'):
            self._tracker.track(tracklets=[], detections=_detections(0.0), frame_index=1, frame=None)

    def test_requires_image_is_derived_from_components(self) -> None:
        """
        `Tracker.requires_image` reflects the configured CMC rather than a YAML flag.
        """
        self.assertFalse(SortTracker(_config(None)).requires_image)
        self.assertFalse(SortTracker(_config({'name': 'identity'})).requires_image)


if __name__ == '__main__':
    unittest.main()
