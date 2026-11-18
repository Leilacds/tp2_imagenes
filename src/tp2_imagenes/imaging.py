from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def normalize_image(image: Image.Image | None) -> Image.Image:
    if image is None:
        raise ValueError("Primero cargá una imagen.")

    rgb_image = image.convert("RGB")
    if min(rgb_image.size) < 64:
        raise ValueError("La imagen debe medir al menos 64 × 64 píxeles.")
    return rgb_image


def prepare_for_diffusion(image: Image.Image, max_side: int = 768) -> Image.Image:
    if max_side < 64:
        raise ValueError("El lado máximo debe ser de al menos 64 píxeles.")

    scale = min(1.0, max_side / max(image.size))
    width = int(image.width * scale) // 8 * 8
    height = int(image.height * scale) // 8 * 8
    if min(width, height) < 64:
        raise ValueError(
            "La proporción de la imagen es demasiado extrema para procesarla."
        )

    if (width, height) == image.size:
        return image.copy()
    return image.resize((width, height), Image.Resampling.LANCZOS)


def bilateral_baseline(image: Image.Image) -> Image.Image:
    rgb_image = np.asarray(image.convert("RGB"))
    filtered = cv2.bilateralFilter(
        rgb_image,
        d=7,
        sigmaColor=50,
        sigmaSpace=50,
    )
    return Image.fromarray(filtered)


def difference_map(
    before: Image.Image,
    after: Image.Image,
    amplification: int = 3,
) -> Image.Image:
    after_aligned = after.convert("RGB").resize(before.size, Image.Resampling.LANCZOS)
    before_array = np.asarray(before.convert("RGB"), dtype=np.int16)
    after_array = np.asarray(after_aligned, dtype=np.int16)
    difference = np.abs(before_array - after_array)
    amplified = np.clip(difference * amplification, 0, 255).astype(np.uint8)
    return Image.fromarray(amplified)
