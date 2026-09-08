from __future__ import annotations

import math

import numpy as np

from .schemas import GarmentEvent


FEATURE_NAMES = [
    "brightness",
    "blur_log",
    "nir_signal",
    "category_confidence",
    "grade_confidence",
    "ood_score",
    "inference_ms",
    "total_latency_ms",
    "defect_count",
    "max_defect_severity",
]


def operational_vector(event: GarmentEvent) -> np.ndarray:
    """Compact feature vector used when the producer does not send embeddings."""
    return np.asarray(
        [
            float(event.frame_quality.brightness),
            math.log1p(max(0.0, float(event.frame_quality.blur_variance))),
            float(event.nir.signal_quality),
            float(event.vision.category_confidence),
            float(event.grade.confidence),
            float(event.vision.ood_score),
            float(event.vision.inference_ms),
            float(event.total_latency_ms),
            float(len(event.vision.defects)),
            float(event.max_defect_severity),
        ],
        dtype=np.float64,
    )


def drift_vector(event: GarmentEvent) -> np.ndarray:
    """Use real embeddings when present, otherwise fall back to operational features."""
    if event.embedding is not None:
        return np.asarray(event.embedding, dtype=np.float64)
    return operational_vector(event)


def scalar_features(event: GarmentEvent) -> dict[str, float]:
    return dict(zip(FEATURE_NAMES, operational_vector(event), strict=True))


def business_risk(event: GarmentEvent) -> float:
    if event.route.requires_human or event.grade.grade == "REVIEW":
        return 1.0
    if event.grade.grade == "C":
        return 0.85
    if event.grade.grade == "B":
        return 0.45
    return 0.20
