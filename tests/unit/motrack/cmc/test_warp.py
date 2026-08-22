"""
Unit tests for the CMC affine warp component.
"""
import unittest

import numpy as np

from motrack.cmc.components.warp import (
    apply_warp_to_points,
    blend_with_identity,
    compose_warps,
    identity_warp,
    invert_warp,
    is_identity_warp,
    normalized_warp_to_pixel,
    pixel_warp_to_normalized
)


def _random_warp(rng: np.random.Generator) -> np.ndarray:
    """
    Creates a random, well-conditioned affine warp close to the identity.
    """
    warp = identity_warp(dtype=np.float64)
    warp[:, :2] += rng.uniform(-0.1, 0.1, size=(2, 2))
    warp[:, 2] = rng.uniform(-20.0, 20.0, size=2)
    return warp


class WarpBasicsTest(unittest.TestCase):
    def test_identity_warp_leaves_points_unchanged(self) -> None:
        """
        Applying the identity warp is a no-op.
        """
        points = np.array([[0.0, 0.0], [1.5, -2.5], [100.0, 42.0]], dtype=np.float32)
        np.testing.assert_array_equal(apply_warp_to_points(identity_warp(), points), points)
        self.assertTrue(is_identity_warp(identity_warp()))

    def test_is_identity_warp_rejects_non_identity(self) -> None:
        """
        A warp with a non-zero translation is not the identity.
        """
        warp = identity_warp()
        warp[0, 2] = 1e-3
        self.assertFalse(is_identity_warp(warp))

    def test_compose_warps_matches_sequential_application(self) -> None:
        """
        Composing two warps equals applying them one after the other.
        """
        rng = np.random.default_rng(0)
        first, second = _random_warp(rng), _random_warp(rng)
        points = rng.uniform(-50.0, 50.0, size=(16, 2))

        expected = apply_warp_to_points(second, apply_warp_to_points(first, points))
        actual = apply_warp_to_points(compose_warps(first, second), points)

        np.testing.assert_allclose(actual, expected, atol=1e-9)

    def test_invert_warp_round_trip(self) -> None:
        """
        Composing a warp with its inverse gives the identity.
        """
        rng = np.random.default_rng(1)
        for _ in range(10):
            warp = _random_warp(rng)
            self.assertTrue(is_identity_warp(compose_warps(warp, invert_warp(warp)), atol=1e-9))

    def test_blend_with_identity_endpoints(self) -> None:
        """
        Blending weight 0 gives the identity and weight 1 gives the original warp.
        """
        warp = _random_warp(np.random.default_rng(2))
        np.testing.assert_allclose(blend_with_identity(warp, 0.0), identity_warp(dtype=np.float64), atol=1e-12)
        np.testing.assert_allclose(blend_with_identity(warp, 1.0), warp, atol=1e-12)

    def test_blend_with_identity_halves_translation(self) -> None:
        """
        Blending weight 0.5 halves a pure translation.
        """
        warp = identity_warp(dtype=np.float64)
        warp[:, 2] = [10.0, -4.0]
        np.testing.assert_allclose(blend_with_identity(warp, 0.5)[:, 2], [5.0, -2.0], atol=1e-12)


class WarpNormalizationTest(unittest.TestCase):
    IMAGE_SIZES = [(1920, 1080), (640, 480), (1000, 100)]

    def test_normalized_warp_transforms_normalized_points(self) -> None:
        """
        Warping normalized points with the normalized warp agrees with warping pixel
        points with the pixel warp and normalizing afterwards.

        This is the defining property of `pixel_warp_to_normalized`.
        """
        rng = np.random.default_rng(3)
        for width, height in self.IMAGE_SIZES:
            with self.subTest(width=width, height=height):
                warp = _random_warp(rng)
                points = np.stack([
                    rng.uniform(0.0, width, size=64),
                    rng.uniform(0.0, height, size=64)
                ], axis=1)
                scale = np.array([width, height], dtype=np.float64)

                expected = apply_warp_to_points(warp, points) / scale
                actual = apply_warp_to_points(pixel_warp_to_normalized(warp, width, height), points / scale)

                np.testing.assert_allclose(actual, expected, atol=1e-9)

    def test_translation_only_rescaling_is_insufficient(self) -> None:
        """
        Rescaling only the translation column - the behaviour this module replaces - does
        not satisfy the property above once the warp contains rotation.

        Regression guard: if this test starts passing, the normalization has silently been
        reduced back to a translation-only rescale.
        """
        width, height = 1920, 1080
        angle = np.deg2rad(5.0)
        warp = np.array([
            [np.cos(angle), -np.sin(angle), 12.0],
            [np.sin(angle), np.cos(angle), -7.0]
        ], dtype=np.float64)

        naive = warp.copy()
        naive[0, 2] /= width
        naive[1, 2] /= height

        points = np.array([[100.0, 200.0], [1500.0, 900.0]], dtype=np.float64)
        scale = np.array([width, height], dtype=np.float64)
        expected = apply_warp_to_points(warp, points) / scale

        self.assertFalse(np.allclose(apply_warp_to_points(naive, points / scale), expected, atol=1e-6))
        np.testing.assert_allclose(
            apply_warp_to_points(pixel_warp_to_normalized(warp, width, height), points / scale),
            expected,
            atol=1e-9
        )

    def test_pixel_normalized_round_trip(self) -> None:
        """
        Converting to normalized coordinates and back recovers the original warp.
        """
        rng = np.random.default_rng(4)
        for width, height in self.IMAGE_SIZES:
            with self.subTest(width=width, height=height):
                for _ in range(20):
                    warp = _random_warp(rng)
                    recovered = normalized_warp_to_pixel(pixel_warp_to_normalized(warp, width, height), width, height)
                    np.testing.assert_allclose(recovered, warp, atol=1e-9)

    def test_translation_and_isotropic_scale_linear_block_unchanged(self) -> None:
        """
        Warps without rotation or shear keep their linear block, so the conversion agrees
        with the legacy translation-only behaviour in exactly those cases.
        """
        width, height = 1920, 1080
        warp = np.array([[1.02, 0.0, 30.0], [0.0, 1.02, -15.0]], dtype=np.float64)
        normalized = pixel_warp_to_normalized(warp, width, height)

        np.testing.assert_allclose(normalized[:, :2], warp[:, :2], atol=1e-12)
        np.testing.assert_allclose(normalized[:, 2], [30.0 / width, -15.0 / height], atol=1e-12)

    def test_off_diagonal_scaled_by_aspect_ratio(self) -> None:
        """
        Pins the direction of the aspect-ratio correction on the off-diagonal terms.
        """
        width, height = 1920, 1080
        warp = np.array([[1.0, 0.3, 0.0], [0.4, 1.0, 0.0]], dtype=np.float64)
        normalized = pixel_warp_to_normalized(warp, width, height)

        self.assertAlmostEqual(normalized[0, 1], 0.3 * height / width, places=12)
        self.assertAlmostEqual(normalized[1, 0], 0.4 * width / height, places=12)

    def test_downscale_invariance(self) -> None:
        """
        The same geometric motion estimated on a downscaled image yields the same
        normalized warp, which is what lets algorithms normalize by the downscaled size.
        """
        width, height, factor = 1920, 1080, 2
        angle = np.deg2rad(3.0)
        linear = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ], dtype=np.float64)

        full = np.concatenate([linear, np.array([[24.0], [-16.0]])], axis=1)
        small = np.concatenate([linear, np.array([[24.0 / factor], [-16.0 / factor]])], axis=1)

        np.testing.assert_allclose(
            pixel_warp_to_normalized(small, width // factor, height // factor),
            pixel_warp_to_normalized(full, width, height),
            atol=1e-12
        )

    def test_preserves_dtype(self) -> None:
        """
        The conversion keeps the input dtype, which matters because the motion filter
        builds float32 matrices.
        """
        warp = identity_warp(dtype=np.float32)
        self.assertEqual(pixel_warp_to_normalized(warp, 1920, 1080).dtype, np.float32)


if __name__ == '__main__':
    unittest.main()
