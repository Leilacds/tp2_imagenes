from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import TYPE_CHECKING

from PIL import Image
from transformers.pipelines import pipeline

if TYPE_CHECKING:
    from transformers.pipelines import Pipeline

MODEL_ID = "PekingU/rtdetr_r50vd"
SCORE_THRESHOLD = 0.4


@lru_cache(maxsize=1)
def _load_detector() -> Pipeline:
    return pipeline(
        task="object-detection",
        model=MODEL_ID,
        device=-1,
    )


def _detected_labels(
    detections: Sequence[Mapping[str, object]],
) -> Counter[str]:
    labels: Counter[str] = Counter()
    for detection in detections:
        label = detection.get("label")
        score = detection.get("score")
        if (
            isinstance(label, str)
            and isinstance(score, (int, float))
            and score >= SCORE_THRESHOLD
        ):
            labels[label] += 1
    return labels


def _format_counts(counts: Counter[str]) -> str:
    if not counts:
        return "ninguno con confianza suficiente"
    return ", ".join(
        f"{label} ({count})"
        for label, count in sorted(counts.items())
    )


def compare_detections(before: Image.Image, after: Image.Image) -> str:
    detector = _load_detector()
    before_counts = _detected_labels(detector(before))
    after_counts = _detected_labels(detector(after))
    added = after_counts - before_counts
    removed = before_counts - after_counts

    changes = []
    if added:
        changes.append(f"- Etiquetas que aparecen después: {_format_counts(added)}.")
    if removed:
        changes.append(f"- Etiquetas que ya no aparecen: {_format_counts(removed)}.")
    if not changes:
        changes.append("- No cambiaron las etiquetas detectadas con el umbral elegido.")

    return "\n".join(
        [
            "### Análisis de objetos (RT-DETR)",
            f"- Antes: {_format_counts(before_counts)}.",
            f"- Después: {_format_counts(after_counts)}.",
            *changes,
            "",
            (
                "Esto compara etiquetas, no la identidad exacta de los objetos ni la "
                "autenticidad de la imagen. El detector también puede equivocarse."
            ),
        ]
    )
