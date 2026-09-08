from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sort_drift_sentinel.core.features import business_risk, drift_vector
from sort_drift_sentinel.core.schemas import ActiveLearningItem, GarmentEvent


@dataclass
class _Candidate:
    event: GarmentEvent
    vector: np.ndarray
    uncertainty: float
    ood: float
    risk: float

    @property
    def base(self) -> float:
        return 0.45 * self.uncertainty + 0.30 * self.ood + 0.25 * self.risk


class ActiveLearningRanker:
    """Greedy uncertainty/OOD/business-risk ranking with a diversity term."""

    def rank(self, events: list[GarmentEvent], k: int = 20) -> list[ActiveLearningItem]:
        if not events or k <= 0:
            return []
        candidates = [
            _Candidate(
                event=e,
                vector=drift_vector(e),
                uncertainty=1.0 - e.min_task_confidence,
                ood=e.vision.ood_score,
                risk=business_risk(e),
            )
            for e in events
        ]
        # Keep vectors of one dimension. In production this should be a named embedding version.
        counts: dict[int, int] = {}
        for c in candidates:
            counts[len(c.vector)] = counts.get(len(c.vector), 0) + 1
        target_dim = max(counts, key=counts.get)
        candidates = [c for c in candidates if len(c.vector) == target_dim]

        selected: list[_Candidate] = []
        output: list[ActiveLearningItem] = []
        remaining = candidates[:]
        scale = self._scale([c.vector for c in candidates])

        while remaining and len(selected) < min(k, len(candidates)):
            best: _Candidate | None = None
            best_score = -1.0
            best_diversity = 0.0
            for candidate in remaining:
                if not selected:
                    diversity = 1.0
                else:
                    distances = [
                        np.linalg.norm((candidate.vector - s.vector) / scale) for s in selected
                    ]
                    diversity = float(np.tanh(min(distances)))
                score = 0.80 * candidate.base + 0.20 * diversity
                if score > best_score:
                    best_score = score
                    best = candidate
                    best_diversity = diversity
            assert best is not None
            selected.append(best)
            remaining.remove(best)
            reasons = []
            if best.uncertainty > 0.30:
                reasons.append("low confidence")
            if best.ood > 0.35:
                reasons.append("OOD")
            if best.risk >= 0.80:
                reasons.append("high business risk")
            if best_diversity > 0.70:
                reasons.append("diverse sample")
            output.append(
                ActiveLearningItem(
                    event_id=best.event.event_id,
                    garment_id=best.event.garment_id,
                    deployment_id=best.event.deployment_id,
                    priority=round(best_score, 4),
                    uncertainty=round(best.uncertainty, 4),
                    ood=round(best.ood, 4),
                    business_risk=round(best.risk, 4),
                    diversity=round(best_diversity, 4),
                    reason=", ".join(reasons) or "representative sample",
                )
            )
        return output

    @staticmethod
    def _scale(vectors: list[np.ndarray]) -> np.ndarray:
        matrix = np.vstack(vectors)
        std = np.std(matrix, axis=0)
        return np.where(std < 1e-6, 1.0, std)
