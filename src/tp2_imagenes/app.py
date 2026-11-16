from __future__ import annotations

import os

import gradio as gr
from PIL import Image

from tp2_imagenes.analysis import compare_detections
from tp2_imagenes.diffusion import denoise_image
from tp2_imagenes.imaging import (
    bilateral_baseline,
    difference_map,
    normalize_image,
)


def process_image(
    image: Image.Image | None,
    strength: float,
    seed: float,
) -> tuple[tuple[Image.Image, Image.Image], Image.Image, Image.Image, str]:
    try:
        original = normalize_image(image)
    except ValueError as error:
        raise gr.Error(str(error)) from error

    try:
        restored = denoise_image(original, strength=strength, seed=int(seed))
    except (RuntimeError, ValueError) as error:
        raise gr.Error(str(error)) from error

    baseline = bilateral_baseline(original)
    changes = difference_map(original, restored)
    report = compare_detections(original, restored)
    return (original, restored), baseline, changes, report


def create_app() -> gr.Blocks:
    with gr.Blocks(title="Reducción de ruido con difusión") as demo:
        gr.Markdown(
            "# Reducción de ruido en fotos con poca luz\n"
            "Subí una foto y compará la difusión con un filtro bilateral clásico. "
            "**La IA puede inventar o cambiar detalles; el resultado no es "
            "evidencia de la escena original.**"
        )

        with gr.Row():
            image_input = gr.Image(type="pil", label="Imagen de entrada")
            with gr.Column():
                strength = gr.Slider(
                    minimum=0.1,
                    maximum=0.45,
                    value=0.25,
                    step=0.05,
                    label="Intensidad de difusión (más alto = más cambios)",
                )
                seed = gr.Slider(
                    minimum=0,
                    maximum=999999,
                    value=42,
                    step=1,
                    label="Semilla (para repetir la configuración)",
                )
                process_button = gr.Button("Reducir ruido", variant="primary")

        comparison = gr.ImageSlider(
            type="pil",
            label="Comparación: original / difusión",
        )
        with gr.Row():
            baseline_output = gr.Image(
                type="pil",
                label="Línea base: filtro bilateral",
            )
            changes_output = gr.Image(
                type="pil",
                label="Mapa de diferencias (no indica si el cambio es correcto)",
            )
        report_output = gr.Markdown(label="Análisis visual")

        process_button.click(
            process_image,
            inputs=[image_input, strength, seed],
            outputs=[comparison, baseline_output, changes_output, report_output],
        )

    return demo


def main() -> None:
    demo = create_app()
    in_colab = "COLAB_RELEASE_TAG" in os.environ
    demo.queue().launch(
        share=in_colab,
        server_name="0.0.0.0" if in_colab else "127.0.0.1",
    )
