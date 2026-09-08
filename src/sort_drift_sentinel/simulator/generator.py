from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from sort_drift_sentinel.core.schemas import GarmentEvent


class GarmentEventGenerator:
    def __init__(self, seed: int = 7) -> None:
        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        deployment_id: str,
        n: int,
        mode: str = "healthy",
        model_version: str = "grader-0.8.4",
        start_index: int = 0,
    ) -> list[GarmentEvent]:
        events: list[GarmentEvent] = []
        now = datetime.now(timezone.utc)
        for i in range(n):
            brightness = self.rng.normal(128, 8)
            blur = max(5, self.rng.normal(150, 25))
            nir_quality = np.clip(self.rng.normal(0.94, 0.025), 0, 1)
            category_conf = np.clip(self.rng.normal(0.93, 0.035), 0, 1)
            grade_conf = np.clip(self.rng.normal(0.90, 0.04), 0, 1)
            ood = np.clip(self.rng.normal(0.08, 0.035), 0, 1)
            inference = max(4, self.rng.normal(21, 3))
            total_latency = inference + max(4, self.rng.normal(13, 2.5))
            requires_human = self.rng.random() < 0.05

            if mode == "lighting_shift":
                brightness += 35
                blur *= 0.72
                category_conf -= 0.12
                grade_conf -= 0.08
                ood += 0.10
                requires_human = self.rng.random() < 0.16
            elif mode == "critical_drift":
                category_conf -= 0.28
                grade_conf -= 0.22
                ood += 0.38
                requires_human = self.rng.random() < 0.42
            elif mode == "runtime_slowdown":
                inference *= 2.2
                total_latency = inference + self.rng.normal(25, 5)
            elif mode == "nir_shift":
                nir_quality -= 0.30
                grade_conf -= 0.08

            category_conf = float(np.clip(category_conf, 0.05, 0.999))
            grade_conf = float(np.clip(grade_conf, 0.05, 0.999))
            ood = float(np.clip(ood, 0.0, 0.999))
            nir_quality = float(np.clip(nir_quality, 0.05, 0.999))

            grade = "REVIEW" if requires_human else self.rng.choice(["A", "B", "C"], p=[0.35, 0.50, 0.15])
            route = "manual_review" if requires_human else {
                "A": "premium_resale",
                "B": "resale",
                "C": "repair_or_recycle",
            }[grade]
            embedding = self.rng.normal(0, 1, 8)
            if mode == "critical_drift":
                embedding[:3] += 1.7
            elif mode == "lighting_shift":
                embedding[0] += 0.7

            events.append(
                GarmentEvent.model_validate(
                    {
                        "event_id": f"{deployment_id}-{start_index+i:05d}",
                        "garment_id": f"garment-{start_index+i:05d}",
                        "customer": deployment_id.split("-")[0],
                        "policy_version": 42,
                        "timestamp": (now + timedelta(seconds=i)).isoformat(),
                        "frame_quality": {
                            "brightness": float(brightness),
                            "blur_variance": float(blur),
                            "valid": bool(40 < brightness < 220 and blur > 25),
                            "reasons": [],
                        },
                        "nir": {
                            "polyester": 0.25,
                            "cotton": 0.55,
                            "wool": 0.10,
                            "other": 0.10,
                            "signal_quality": nir_quality,
                        },
                        "vision": {
                            "category": "t-shirt",
                            "category_confidence": category_conf,
                            "brand_top5": [["demo-brand", max(0.2, category_conf - 0.03)]],
                            "attributes": {
                                "color": ["black", max(0.2, category_conf - 0.05)],
                                "sleeve": ["short", max(0.2, category_conf - 0.04)],
                            },
                            "defects": [],
                            "embedding_norm": float(np.linalg.norm(embedding)),
                            "ood_score": ood,
                            "model_version": model_version,
                            "inference_ms": float(inference),
                        },
                        "grade": {
                            "grade": grade,
                            "confidence": grade_conf,
                            "reason": "simulated",
                        },
                        "route": {
                            "route": route,
                            "reason": "simulated",
                            "requires_human": requires_human,
                        },
                        "total_latency_ms": float(total_latency),
                        "metadata": {
                            "deployment_id": deployment_id,
                            "camera_id": f"rgb-{deployment_id[-2:]}",
                            "taxonomy_version": "taxonomy-v42",
                            "embedding": embedding.tolist(),
                        },
                    }
                )
            )
        return events
