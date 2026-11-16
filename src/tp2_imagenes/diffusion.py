from __future__ import annotations

from functools import lru_cache

import torch
from diffusers import AutoPipelineForImage2Image
from PIL import Image

from tp2_imagenes.imaging import prepare_for_diffusion

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
PROMPT = (
    "A natural, clear photograph of the same scene. Preserve the original "
    "composition, objects, colors, and lighting. Reduce visible image noise."
)
NEGATIVE_PROMPT = (
    "new objects, missing objects, changed composition, changed text, "
    "extra details, artifacts, illustration, painting"
)


@lru_cache(maxsize=1)
def _load_pipeline() -> AutoPipelineForImage2Image:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No se detectó una GPU CUDA. Ejecutá la aplicación en Google Colab "
            "con una GPU habilitada."
        )

    pipeline = AutoPipelineForImage2Image.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipeline.enable_vae_slicing()
    pipeline.enable_model_cpu_offload()
    return pipeline


def denoise_image(
    image: Image.Image,
    strength: float = 0.25,
    seed: int = 42,
) -> Image.Image:
    if not 0.1 <= strength <= 0.45:
        raise ValueError("La intensidad debe estar entre 0.10 y 0.45.")

    pipeline = _load_pipeline()
    prepared_image = prepare_for_diffusion(image)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    result = pipeline(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        image=prepared_image,
        strength=strength,
        guidance_scale=4.0,
        num_inference_steps=20,
        generator=generator,
    ).images[0]
    return result.resize(image.size, Image.Resampling.LANCZOS)
