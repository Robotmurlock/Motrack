"""
Unit tests for the custom pyramidal Lucas-Kanade optical flow estimator.

Coordinate convention encoded by these tests: **pixels in, pixels out**. Feature points are
given in pixel coordinates of the full resolution frame and the returned flow is a pixel
displacement. This is the conventional Lucas-Kanade contract, it matches what
`cv2.calcOpticalFlowPyrLK` does, and it is what the CMC pipeline needs - correspondences are
handed to `WarpRANSACEstimator` together with a residual threshold in (downscaled) pixels.
"""
import unittest

import cv2
import numpy as np

from motrack.cmc.components.pylk import PyLucasKanadeEstimator

IMAGE_HEIGHT, IMAGE_WIDTH = 240, 320


def _textured_gray(seed: int = 0, height: int = IMAGE_HEIGHT, width: int = IMAGE_WIDTH) -> np.ndarray:
    """
    Builds a deterministic, well-textured grayscale image.

    Optical flow is only defined where the image has structure, so the noise is smoothed into
    blobs with real gradients rather than being left as per-pixel noise.
    """
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, size=(height, width)).astype(np.float32)
    blurred = cv2.GaussianBlur(noise, (0, 0), sigmaX=2.0)
    normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)


def _as_rgb(gray: np.ndarray) -> np.ndarray:
    """
    Replicates a grayscale image into three channels.

    `estimate` currently calls `cv2.cvtColor(..., COLOR_RGB2GRAY)` itself, so it requires
    three channel input.
    """
    return np.repeat(gray[:, :, None], 3, axis=2)


def _textured_image(seed: int = 0) -> np.ndarray:
    """
    Builds the RGB frame the estimator currently expects.
    """
    return _as_rgb(_textured_gray(seed))


def _shift_image(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Translates an image by a (possibly fractional) offset.

    Args:
        image: Source image
        dx: Horizontal shift in pixels
        dy: Vertical shift in pixels

    Returns:
        Shifted image
    """
    warp = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(image, warp, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _interior_points(margin: int = 60, step: int = 40) -> np.ndarray:
    """
    Builds a grid of feature points well away from the image border.
    """
    xs = np.arange(margin, IMAGE_WIDTH - margin, step, dtype=np.float32)
    ys = np.arange(margin, IMAGE_HEIGHT - margin, step, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float32)


def _opencv_reference(prev_frame: np.ndarray, next_frame: np.ndarray, points: np.ndarray, window: int, max_level: int) -> np.ndarray:
    """
    Computes the same flow with OpenCV, used as a correctness reference.
    """
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(
        cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY), cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY),
        points.reshape(-1, 1, 2).astype(np.float32), None,
        winSize=(window, window), maxLevel=max_level,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )
    flows = tracked.reshape(-1, 2) - points
    flows[status.ravel() == 0] = np.nan
    return flows


class PyLucasKanadeSmokeTest(unittest.TestCase):
    """
    The estimator must run at all on the inputs the pipeline provides.
    """

    def test_runs_on_rgb_frames(self) -> None:
        """
        `estimate` converts RGB to grayscale itself, so three channel frames must work.
        """
        prev_frame = _textured_image()
        next_frame = _shift_image(prev_frame, 2.0, 1.0)

        flows, _ = PyLucasKanadeEstimator().estimate(prev_frame, next_frame, _interior_points())

        self.assertEqual(flows.shape, (_interior_points().shape[0], 2))
        self.assertTrue(np.isfinite(flows).all())

    def test_accepts_grayscale_frames(self) -> None:
        """
        The CMC pipeline converts to grayscale once, in `feature_cmc`, and hands single
        channel frames to the correspondence source. Converting again inside the estimator
        means it cannot consume what the pipeline produces.
        """
        prev_frame = _textured_gray()
        next_frame = _shift_image(prev_frame, 2.0, 1.0)

        flows, _ = PyLucasKanadeEstimator().estimate(prev_frame, next_frame, _interior_points())

        self.assertEqual(flows.shape, (_interior_points().shape[0], 2))

    def test_zero_motion_gives_zero_flow(self) -> None:
        """
        Identical frames must produce no displacement.
        """
        frame = _textured_image()
        flows, skips = PyLucasKanadeEstimator().estimate(frame, frame.copy(), _interior_points())

        self.assertFalse(skips.any(), 'Zero flow must mean tracked-and-stationary, not skipped')
        np.testing.assert_allclose(flows, 0.0, atol=0.1)


class PyLucasKanadeAccuracyTest(unittest.TestCase):
    """
    Recovery of known translations.
    """

    def test_recovers_integer_translation(self) -> None:
        """
        A whole-pixel shift is the easiest possible case.
        """
        prev_frame = _textured_image()
        dx, dy = 4.0, -3.0
        next_frame = _shift_image(prev_frame, dx, dy)

        flows, skips = PyLucasKanadeEstimator().estimate(prev_frame, next_frame, _interior_points())

        self.assertFalse(skips.any(), 'Well textured interior points must all be trackable')
        np.testing.assert_allclose(np.median(flows, axis=0), [dx, dy], atol=0.5)

    def test_recovers_subpixel_translation(self) -> None:
        """
        The refinement loop exists to reach sub-pixel accuracy.

        This is unreachable with integer patch sampling: a correction smaller than one pixel
        leaves the sampled patch bit-identical, so the residual and the correction never
        change and the iteration stalls.
        """
        prev_frame = _textured_image()
        dx, dy = 3.5, -2.25
        next_frame = _shift_image(prev_frame, dx, dy)

        flows, skips = PyLucasKanadeEstimator().estimate(prev_frame, next_frame, _interior_points())

        self.assertFalse(skips.any(), 'Well textured interior points must all be trackable')
        np.testing.assert_allclose(np.median(flows, axis=0), [dx, dy], atol=0.2)

    def test_recovers_large_translation_via_pyramid(self) -> None:
        """
        A shift beyond the reach of a single level is recovered only coarse to fine.

        Both frames are cropped from a larger source rather than warped, so no synthetic
        border is introduced: a reflected band would otherwise dominate the coarse levels,
        where a 21 pixel window spans many times its own width of the original image.

        16 pixels is chosen because it is the point where the pyramid demonstrably earns its
        keep - a single level cannot find it, four levels recover it exactly. Larger shifts
        are not asserted: on this synthetic texture 26 pixels defeats OpenCV as well, so it
        measures the test image rather than the implementation.
        """
        source = _textured_gray(seed=1, height=IMAGE_HEIGHT + 80, width=IMAGE_WIDTH + 80)
        dx, dy = 16.0, 0.0
        prev_frame = _as_rgb(source[40:40 + IMAGE_HEIGHT, 40:40 + IMAGE_WIDTH])
        next_frame = _as_rgb(source[40:40 + IMAGE_HEIGHT, 40 + int(dx):40 + int(dx) + IMAGE_WIDTH])
        points = _interior_points(margin=80)

        single, _ = PyLucasKanadeEstimator(window_size=21, max_level=1).estimate(prev_frame, next_frame, points)
        pyramid, skips = PyLucasKanadeEstimator(window_size=21, max_level=4).estimate(prev_frame, next_frame, points)

        # Cropping forward by dx makes the content move backwards by dx.
        self.assertFalse(skips.any())
        self.assertGreater(abs(np.median(single, axis=0)[0] + dx), 1.0, 'A single level should not reach this shift')
        np.testing.assert_allclose(np.median(pyramid, axis=0), [-dx, dy], atol=0.5)

    def test_flow_is_in_pixel_units(self) -> None:
        """
        A 10 pixel shift must produce a flow of magnitude 10, not 10/width.
        """
        prev_frame = _textured_image()
        next_frame = _shift_image(prev_frame, 10.0, 0.0)

        flows, skips = PyLucasKanadeEstimator().estimate(prev_frame, next_frame, _interior_points())

        self.assertFalse(skips.any(), 'Well textured interior points must all be trackable')
        self.assertAlmostEqual(float(np.median(flows[:, 0])), 10.0, delta=1.0)

    def test_agrees_with_opencv(self) -> None:
        """
        Cross-check against the reference implementation the class is modelled on.
        """
        prev_frame = _textured_image()
        next_frame = _shift_image(prev_frame, 5.0, -4.0)
        points = _interior_points()

        expected = _opencv_reference(prev_frame, next_frame, points, window=21, max_level=3)
        actual, _ = PyLucasKanadeEstimator(window_size=21, max_level=3).estimate(prev_frame, next_frame, points)

        valid = np.isfinite(expected).all(axis=1)
        np.testing.assert_allclose(actual[valid], expected[valid], atol=1.0)


class PyLucasKanadePyramidTest(unittest.TestCase):
    """
    Structure of the pyramid itself.
    """

    def test_scale_frame_to_level_halves_per_level(self) -> None:
        """
        Each level is half the resolution of the previous one, measured from the original.
        """
        estimator = PyLucasKanadeEstimator()
        frame = _textured_gray()

        for level in range(4):
            with self.subTest(level=level):
                scaled = estimator._scale_frame_to_level(frame, level)  # pylint: disable=protected-access
                self.assertEqual(scaled.shape, (IMAGE_HEIGHT // 2 ** level, IMAGE_WIDTH // 2 ** level))

    def test_pyramid_is_built_from_the_original_frame_coarse_to_fine(self) -> None:
        """
        Levels must be visited coarse to fine, and each must be derived from the original
        frame rather than from the previously scaled one.

        The scaled frames must also actually reach the per-point solver: computing them and
        then passing the full resolution frames on makes the pyramid a no-op.
        """
        estimator = PyLucasKanadeEstimator(max_level=3)
        frame = _textured_image()

        scale_calls, flow_shapes = [], []
        original_scale = estimator._scale_frame_to_level  # pylint: disable=protected-access
        original_compute = estimator._compute_flow  # pylint: disable=protected-access

        def _scale_spy(source: np.ndarray, level: int) -> np.ndarray:
            scale_calls.append((source.shape[:2], level))
            return original_scale(source, level)

        def _compute_spy(*args, **kwargs) -> np.ndarray:
            frame = kwargs['prev_level_frame'] if 'prev_level_frame' in kwargs else args[0]
            flow_shapes.append(frame.shape[:2])
            return original_compute(*args, **kwargs)

        estimator._scale_frame_to_level = _scale_spy  # pylint: disable=protected-access
        estimator._compute_flow = _compute_spy  # pylint: disable=protected-access
        estimator.estimate(frame, _shift_image(frame, 2.0, 0.0), _interior_points())

        levels = [level for _, level in scale_calls]
        self.assertEqual(levels, sorted(levels, reverse=True), 'Levels must be visited coarse to fine')
        self.assertTrue(
            all(shape == (IMAGE_HEIGHT, IMAGE_WIDTH) for shape, _ in scale_calls),
            'Every level must be scaled from the original frame, not from the previous level'
        )
        self.assertGreater(
            len(set(flow_shapes)), 1,
            'The solver always saw the same resolution, so the scaled frames never reached it'
        )


class PyLucasKanadeRobustnessTest(unittest.TestCase):
    """
    Degenerate inputs must not raise: the CMC contract forbids it.
    """

    def test_points_near_the_border_do_not_crash(self) -> None:
        """
        `goodFeaturesToTrack` routinely returns points close to the image border, where the
        patch window extends past the edge.
        """
        prev_frame = _textured_image()
        next_frame = _shift_image(prev_frame, 2.0, 2.0)
        points = np.array([[1.0, 1.0], [IMAGE_WIDTH - 2.0, IMAGE_HEIGHT - 2.0], [0.0, 0.0]], dtype=np.float32)

        flows, _ = PyLucasKanadeEstimator().estimate(prev_frame, next_frame, points)

        self.assertEqual(flows.shape, (3, 2))

    def test_uniform_region_does_not_raise(self) -> None:
        """
        A textureless patch makes the 2x2 gradient matrix singular, so `np.linalg.solve`
        raises rather than returning a poor estimate.
        """
        prev_frame = _as_rgb(np.full((IMAGE_HEIGHT, IMAGE_WIDTH), 128, dtype=np.uint8))
        next_frame = prev_frame.copy()
        points = np.array([[160.0, 120.0]], dtype=np.float32)

        flows, _ = PyLucasKanadeEstimator().estimate(prev_frame, next_frame, points)

        self.assertEqual(flows.shape, (1, 2))

    def test_intensity_residual_does_not_wrap_around(self) -> None:
        """
        The residual `patch_q - patch_p` must be computed in a signed type.

        On raw uint8 patches a negative difference wraps (5 - 10 becomes 251), which turns
        every darkening pixel into a large positive residual and corrupts the solve. A
        strong bright-to-dark gradient makes the effect unmissable.
        """
        # Texture modulated by a strong dark-to-bright ramp: the ramp makes the residual
        # sign-sensitive, the texture keeps both gradient directions non-zero so the point
        # is not rejected by the eigenvalue guard before the residual is ever used.
        ramp = np.linspace(0.2, 1.0, IMAGE_WIDTH, dtype=np.float32)[None, :]
        modulated = (_textured_gray().astype(np.float32) * ramp).astype(np.uint8)
        prev_frame = _as_rgb(modulated)
        next_frame = _shift_image(prev_frame, 3.0, 0.0)
        points = _interior_points()

        flows, skips = PyLucasKanadeEstimator().estimate(prev_frame, next_frame, points)

        self.assertFalse(skips.any(), 'Points must be trackable, otherwise the residual is never exercised')
        self.assertAlmostEqual(float(np.median(flows[:, 0])), 3.0, delta=1.0)


OBJECT_SIZE = 60
SUPERSAMPLE = 4
OBJECT_CENTRES = np.array([[80.0, 70.0], [210.0, 80.0], [140.0, 170.0]], dtype=np.float32)


def _render_scene(offsets: np.ndarray) -> np.ndarray:
    """
    Renders three textured objects on a black background, each displaced by its own offset.

    Both frames of a pair are rendered independently rather than warped, so there is no
    synthetic border anywhere and the ground truth displacement is exact. Rendering happens
    at `SUPERSAMPLE` times the final resolution, which makes fractional displacements real
    image content rather than an interpolation of the finished frame.

    Args:
        offsets: Per-object (dx, dy) displacements, shape (3, 2)

    Returns:
        RGB frame
    """
    canvas = np.zeros((IMAGE_HEIGHT * SUPERSAMPLE, IMAGE_WIDTH * SUPERSAMPLE), dtype=np.uint8)
    side = OBJECT_SIZE * SUPERSAMPLE

    for index, ((centre_x, centre_y), (dx, dy)) in enumerate(zip(OBJECT_CENTRES, offsets)):
        rng = np.random.default_rng(index)
        texture = cv2.normalize(
            cv2.GaussianBlur(rng.integers(0, 256, (OBJECT_SIZE, OBJECT_SIZE)).astype(np.float32), (0, 0), 1.5),
            None, 40, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        upscaled = cv2.resize(texture, (side, side), interpolation=cv2.INTER_NEAREST)
        x = int(round((centre_x + dx) * SUPERSAMPLE)) - side // 2
        y = int(round((centre_y + dy) * SUPERSAMPLE)) - side // 2
        canvas[y:y + side, x:x + side] = upscaled

    return _as_rgb(cv2.resize(canvas, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA))


class PyLucasKanadeSceneTest(unittest.TestCase):
    """
    End to end tests on an explicit synthetic scene: three textured objects on a black
    background, tracked at their centres.

    This is the closest analogue to the CMC use case. A rigid displacement of the whole
    scene is what camera motion looks like, the black background is untrackable and must be
    reported as such, and independently moving objects are exactly the foreground that
    masking exists to exclude.
    """

    STATIC = np.zeros((3, 2), dtype=np.float32)

    def test_recovers_rigid_scene_displacement(self) -> None:
        """
        Every object displaced by the same amount, which is what camera motion produces.
        """
        for dx, dy in [(3.0, 0.0), (0.0, -4.0), (-6.0, 2.0), (5.0, 7.0), (2.5, -1.5), (0.25, 0.75)]:
            with self.subTest(dx=dx, dy=dy):
                offsets = np.tile(np.array([dx, dy], dtype=np.float32), (3, 1))
                flows, skips = PyLucasKanadeEstimator(max_level=3).estimate(
                    _render_scene(self.STATIC), _render_scene(offsets), OBJECT_CENTRES
                )

                self.assertFalse(skips.any(), 'Textured objects must be trackable')
                np.testing.assert_allclose(flows, [[dx, dy]] * 3, atol=0.2)

    def test_recovers_independent_object_motion(self) -> None:
        """
        Each object followed separately when they move differently.

        Camera motion compensation depends on this being possible: independently moving
        foreground produces correspondences that disagree with the global motion, which is
        what RANSAC has to reject and what detection masking removes up front.
        """
        offsets = np.array([[4.0, 0.0], [-3.0, 2.0], [0.0, 6.0]], dtype=np.float32)

        flows, skips = PyLucasKanadeEstimator(max_level=3).estimate(
            _render_scene(self.STATIC), _render_scene(offsets), OBJECT_CENTRES
        )

        self.assertFalse(skips.any())
        np.testing.assert_allclose(flows, offsets, atol=0.2)

    def test_background_points_are_reported_untrackable(self) -> None:
        """
        The black background carries no gradient, so points there cannot be tracked.
        """
        offsets = np.tile(np.array([3.0, 0.0], dtype=np.float32), (3, 1))
        background = np.array([[20.0, 220.0], [300.0, 30.0]], dtype=np.float32)

        _, skips = PyLucasKanadeEstimator(max_level=3).estimate(
            _render_scene(self.STATIC), _render_scene(offsets), background
        )

        self.assertTrue(skips.all())

    def test_mixed_object_and_background_points(self) -> None:
        """
        Object and background points in one call keep their positions in the output.
        """
        offsets = np.tile(np.array([3.0, -2.0], dtype=np.float32), (3, 1))
        points = np.concatenate([OBJECT_CENTRES, np.array([[20.0, 220.0], [300.0, 30.0]], dtype=np.float32)])

        flows, skips = PyLucasKanadeEstimator(max_level=3).estimate(
            _render_scene(self.STATIC), _render_scene(offsets), points
        )

        self.assertEqual(flows.shape, (len(points), 2))
        np.testing.assert_array_equal(skips, [False, False, False, True, True])
        np.testing.assert_allclose(flows[:3], [[3.0, -2.0]] * 3, atol=0.2)


class PyLucasKanadeValidityTest(unittest.TestCase):
    """
    Tests for the per-point validity output.

    `feature_cmc` filters correspondences on this flag before handing them to the affine
    estimator, so its polarity and alignment matter as much as the flow itself. A flow of
    zero is ambiguous on its own - it means either "tracked, and did not move" or "never
    tracked at all" - and only this flag distinguishes them.
    """

    def test_trackable_points_are_not_skipped(self) -> None:
        """
        Well textured interior points must all come back valid.
        """
        prev_frame = _textured_image()
        next_frame = _shift_image(prev_frame, 3.0, -2.0)
        points = _interior_points()

        _, skips = PyLucasKanadeEstimator().estimate(prev_frame, next_frame, points)

        self.assertEqual(skips.dtype, bool)
        self.assertFalse(skips.any())

    def test_untrackable_points_are_skipped(self) -> None:
        """
        A textureless region has a singular gradient matrix, so its points are not trackable.

        The flag must say so rather than reporting a confident zero flow.
        """
        prev_frame = _as_rgb(np.full((IMAGE_HEIGHT, IMAGE_WIDTH), 128, dtype=np.uint8))
        next_frame = _shift_image(prev_frame, 3.0, -2.0)
        points = _interior_points()

        _, skips = PyLucasKanadeEstimator().estimate(prev_frame, next_frame, points)

        self.assertTrue(skips.all(), 'A uniform image offers nothing to track')

    def test_validity_is_aligned_with_the_input_points(self) -> None:
        """
        Trackable and untrackable points must be distinguished individually.

        The left half of the frame is textured and the right half is flat, so the flag has
        to vary across the point set rather than being all-or-nothing.
        """
        gray = _textured_gray()
        gray[:, IMAGE_WIDTH // 2:] = 128
        prev_frame = _as_rgb(gray)
        next_frame = _shift_image(prev_frame, 3.0, 0.0)

        textured = np.array([[80.0, 100.0], [100.0, 140.0]], dtype=np.float32)
        flat = np.array([[240.0, 100.0], [270.0, 140.0]], dtype=np.float32)
        points = np.concatenate([textured, flat])

        flows, skips = PyLucasKanadeEstimator().estimate(prev_frame, next_frame, points)

        self.assertEqual(skips.shape, (len(points),))
        self.assertEqual(flows.shape, (len(points), 2))
        self.assertFalse(skips[:len(textured)].any(), 'Textured half must be trackable')
        self.assertTrue(skips[len(textured):].all(), 'Flat half must be reported untrackable')

    def test_outputs_are_the_same_length_as_the_input(self) -> None:
        """
        Correspondences are formed by pairing input points with output flows, so dropping
        rows silently destroys that pairing. Points that cannot be tracked must be reported
        through the flag, not by shortening the result.
        """
        prev_frame = _textured_image()
        next_frame = _shift_image(prev_frame, 2.0, 2.0)
        points = np.array([
            [120.0, 100.0],
            [1.0, 1.0],
            [160.0, 130.0],
            [IMAGE_WIDTH - 2.0, IMAGE_HEIGHT - 2.0],
            [200.0, 90.0]
        ], dtype=np.float32)

        flows, skips = PyLucasKanadeEstimator().estimate(prev_frame, next_frame, points)

        self.assertEqual(flows.shape, (len(points), 2))
        self.assertEqual(skips.shape, (len(points),))

    def test_skipped_points_do_not_report_a_flow(self) -> None:
        """
        A skipped point keeps its initial zero flow, so a caller that ignores the flag reads
        it as "no motion" rather than as garbage.
        """
        prev_frame = _as_rgb(np.full((IMAGE_HEIGHT, IMAGE_WIDTH), 128, dtype=np.uint8))
        next_frame = _shift_image(prev_frame, 5.0, 5.0)
        points = _interior_points()

        flows, skips = PyLucasKanadeEstimator().estimate(prev_frame, next_frame, points)

        np.testing.assert_array_equal(flows[skips], 0.0)


if __name__ == '__main__':
    unittest.main()
