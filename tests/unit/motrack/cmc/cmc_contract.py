"""
Shared contract every camera motion compensation algorithm has to satisfy, plus the scene
rendering helpers the algorithm level tests are built on.

The contract is defined once here and mixed into a test case per algorithm, so a new
algorithm is held to the same rules without restating them, and a change to the contract is
made in one place. It covers only what `motrack/cmc/algorithms/base.py` mandates - accuracy
is the individual algorithm's business.

This module is deliberately not named `test_*.py`: it holds no tests of its own.
"""
from typing import List, Optional

import cv2
import numpy as np

from motrack.cmc.algorithms.base import CMCContext
from motrack.cmc.catalog import CMC_CATALOG
from motrack.cmc.components.warp import is_identity_warp
from motrack.cmc.factory import cmc_factory
from motrack.library.cv.bbox import BBox, PredBBox

IMAGE_HEIGHT, IMAGE_WIDTH = 360, 480
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
SUPERSAMPLE = 4

OBJECT_CENTRES = np.array([[0.25, 0.30], [0.72, 0.28], [0.45, 0.72]], dtype=np.float32)
OBJECT_SIZE = 70


def render_scene(offsets_px: np.ndarray, background_shift_px: tuple = (0.0, 0.0), seed: int = 7) -> np.ndarray:
    """
    Renders a textured background plus three textured objects, each at its own offset.

    Every frame is rendered independently rather than warped from another, so displacements
    are exact and no synthetic border is introduced - a reflected or replicated border would
    otherwise dominate the coarse pyramid levels. Rendering happens at `SUPERSAMPLE` times the
    final resolution, which makes fractional offsets real image content rather than an
    interpolation of a finished frame.

    Args:
        offsets_px: Per-object (dx, dy) pixel displacements, shape (3, 2)
        background_shift_px: Displacement applied to the background texture
        seed: Seed for the background texture

    Returns:
        RGB frame
    """
    height, width = IMAGE_HEIGHT * SUPERSAMPLE, IMAGE_WIDTH * SUPERSAMPLE
    margin = 64 * SUPERSAMPLE

    rng = np.random.default_rng(seed)
    source = cv2.normalize(
        cv2.GaussianBlur(rng.integers(0, 130, (height + 2 * margin, width + 2 * margin)).astype(np.float32), (0, 0), 5.0),
        None, 0, 150, cv2.NORM_MINMAX
    ).astype(np.uint8)

    offset_x = margin + int(round(background_shift_px[0] * SUPERSAMPLE))
    offset_y = margin + int(round(background_shift_px[1] * SUPERSAMPLE))
    canvas = source[offset_y:offset_y + height, offset_x:offset_x + width].copy()

    side = OBJECT_SIZE * SUPERSAMPLE
    for index, ((centre_x, centre_y), (dx, dy)) in enumerate(zip(OBJECT_CENTRES, offsets_px)):
        texture_rng = np.random.default_rng(index)
        # Rendered at the supersampled resolution and smoothed, rather than upscaled from a
        # small texture with nearest-neighbour. Blocky upscaling produces hard step edges,
        # and FAST fires on those: it put 43% of ORB's features on objects covering 8% of the
        # frame, leaving no background majority for RANSAC to find. Objects stay somewhat
        # corner-dense relative to their area, which is what real pedestrians do too.
        texture = cv2.normalize(
            cv2.GaussianBlur(texture_rng.integers(0, 256, (side, side)).astype(np.float32), (0, 0), 2.0),
            None, 160, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        x = int(round((centre_x * IMAGE_WIDTH + dx) * SUPERSAMPLE)) - side // 2
        y = int(round((centre_y * IMAGE_HEIGHT + dy) * SUPERSAMPLE)) - side // 2
        canvas[y:y + side, x:x + side] = texture

    small = cv2.resize(canvas, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
    return np.repeat(small[:, :, None], 3, axis=2)


def static_offsets() -> np.ndarray:
    """
    Per-object offsets leaving every object where it is.
    """
    return np.zeros((3, 2), dtype=np.float32)


def render_rigid_scene(shift_px: tuple) -> np.ndarray:
    """
    Renders the scene with background and objects displaced together.

    This is what camera motion looks like: the whole image translates. Moving only the
    background would leave the objects as a static minority, and correspondences on them sit
    within the estimator's residual threshold of the true motion for small shifts, so the two
    populations stop being separable and the recovered warp is dragged toward zero.

    Args:
        shift_px: Apparent (dx, dy) displacement of the scene content, in pixels

    Returns:
        RGB frame
    """
    dx, dy = shift_px
    # A positive background shift crops forward, so content appears displaced by -dx. The
    # objects are positioned directly, so they take the negated offset to move with it.
    objects = np.tile(np.array([-dx, -dy], dtype=np.float32), (3, 1))
    return render_scene(objects, background_shift_px=(dx, dy))


def object_detections() -> List[PredBBox]:
    """
    Detections covering the three rendered objects, in normalized coordinates.
    """
    half_w = OBJECT_SIZE / (2 * IMAGE_WIDTH)
    half_h = OBJECT_SIZE / (2 * IMAGE_HEIGHT)
    return [
        PredBBox.create(
            bbox=BBox.from_xyxy(cx - half_w, cy - half_h, cx + half_w, cy + half_h, clip=True),
            label=0,
            conf=0.9
        )
        for cx, cy in OBJECT_CENTRES
    ]


def context(
    frame_index: int,
    prev_frame: Optional[np.ndarray] = None,
    curr_frame: Optional[np.ndarray] = None,
    detections: Optional[List[PredBBox]] = None
) -> CMCContext:
    """
    Builds a context the way the tracker does.
    """
    return CMCContext(
        frame_index=frame_index,
        scene='synthetic',
        prev_frame=prev_frame,
        curr_frame=curr_frame,
        image_size=IMAGE_SIZE,
        detections=detections,
        tracklet_bbox_predictions=[]
    )


def blank_frame(value: int = 128) -> np.ndarray:
    """
    A frame with no texture at all, so nothing is trackable or matchable.
    """
    return np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), value, dtype=np.uint8)


class CMCContractTestMixin:
    """
    Rules from `motrack/cmc/algorithms/base.py` that hold for every algorithm.

    Mix into a `unittest.TestCase` and set `CATALOG_KEY`. Override `build_cmc` when the
    algorithm needs non-default params.
    """
    CATALOG_KEY: str = ''

    def build_cmc(self):
        """
        Creates the algorithm under test through the factory, as a tracker would.
        """
        return cmc_factory(self.CATALOG_KEY, {})

    def test_is_registered_in_the_catalog(self) -> None:
        """
        An algorithm is only reachable from a tracker config once it is registered, under
        matching runtime and config keys.
        """
        self.assertIn(self.CATALOG_KEY, CMC_CATALOG.keys)
        self.assertIn(self.CATALOG_KEY, CMC_CATALOG.config_keys)

    def test_rejects_unknown_params(self) -> None:
        """
        Configs forbid extra fields, so a typo in a YAML key fails loudly instead of being
        silently ignored.
        """
        with self.assertRaisesRegex(ValueError, 'Invalid cmc'):
            CMC_CATALOG.create_config(self.CATALOG_KEY, {'definitely_not_a_field': 1}, params_label='cmc')

    def test_requires_image_is_declared(self) -> None:
        """
        The tracker reads this to fail fast when image loading is disabled.
        """
        self.assertIsInstance(self.build_cmc().requires_image, bool)

    def test_returns_identity_without_a_previous_frame(self) -> None:
        """
        The warp maps frame t-1 into t, so it is undefined without a previous frame.

        The tracker signals both the first frame of a scene and a gap in the sequence by
        leaving `prev_frame` unset, so this single case covers both.
        """
        warp = self.build_cmc().apply(context(frame_index=0, curr_frame=render_scene(static_offsets())))

        self.assertEqual(warp.shape, (2, 3))
        self.assertTrue(is_identity_warp(warp))

    def test_never_raises_on_untextured_frames(self) -> None:
        """
        An algorithm that throws mid-run costs a whole benchmark sweep, so failure is
        reported as an identity warp instead.
        """
        warp = self.build_cmc().apply(
            context(frame_index=1, prev_frame=blank_frame(), curr_frame=blank_frame(140))
        )

        self.assertEqual(warp.shape, (2, 3))
        self.assertTrue(is_identity_warp(warp))

    def test_warp_is_a_2x3_float32_matrix(self) -> None:
        """
        The motion filter builds float32 matrices, so the warp must not widen them.
        """
        prev = render_scene(static_offsets())
        curr = render_scene(static_offsets(), background_shift_px=(3.0, 2.0))
        warp = self.build_cmc().apply(context(frame_index=1, prev_frame=prev, curr_frame=curr))

        self.assertEqual(warp.shape, (2, 3))
        self.assertEqual(warp.dtype, np.float32)

    def test_reset_is_callable(self) -> None:
        """
        The tracker calls this before every scene, whether or not the algorithm has state.
        """
        cmc = self.build_cmc()
        cmc.reset()
        warp = cmc.apply(context(frame_index=0, curr_frame=render_scene(static_offsets())))
        self.assertTrue(is_identity_warp(warp))
