import unittest

import numpy as np
from PIL import Image

from tp2_imagenes.imaging import (
    bilateral_baseline,
    difference_map,
    normalize_image,
    prepare_for_diffusion,
)


class ImagingTests(unittest.TestCase):
    def test_normalize_converts_to_rgb_and_rejects_small_images(self) -> None:
        grayscale = Image.new("L", (64, 64), color=128)
        self.assertEqual(normalize_image(grayscale).mode, "RGB")

        with self.assertRaisesRegex(ValueError, "64"):
            normalize_image(Image.new("RGB", (32, 64)))

    def test_prepare_for_diffusion_keeps_size_and_limits_long_side(self) -> None:
        small = Image.new("RGB", (512, 512))
        self.assertEqual(prepare_for_diffusion(small).size, (512, 512))

        large = Image.new("RGB", (1600, 900))
        prepared = prepare_for_diffusion(large)
        self.assertEqual(prepared.size, (768, 432))

        with self.assertRaisesRegex(ValueError, "proporción"):
            prepare_for_diffusion(Image.new("RGB", (64, 2000)))

    def test_bilateral_baseline_reduces_noise_and_preserves_dimensions(self) -> None:
        generator = np.random.default_rng(42)
        noisy = np.clip(
            generator.normal(128, 20, size=(80, 120, 3)),
            0,
            255,
        ).astype(np.uint8)
        image = Image.fromarray(noisy)
        filtered = np.asarray(bilateral_baseline(image), dtype=np.float32)

        self.assertEqual(filtered.shape, noisy.shape)
        self.assertLess(float(filtered.std()), float(noisy.std()))

    def test_difference_map_is_black_for_identical_images(self) -> None:
        image = Image.new("RGB", (64, 64), color=(30, 60, 90))
        difference = np.asarray(difference_map(image, image))
        self.assertFalse(difference.any())

    def test_difference_map_highlights_changed_pixels(self) -> None:
        before = Image.new("RGB", (64, 64), color="black")
        after = Image.new("RGB", (64, 64), color="black")
        after.putpixel((10, 10), (100, 0, 0))

        difference = np.asarray(difference_map(before, after))
        self.assertTrue(difference[10, 10].any())
        self.assertFalse(difference[0, 0].any())


if __name__ == "__main__":
    unittest.main()
