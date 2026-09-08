from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from sort_drift_sentinel.core.features import drift_vector, scalar_features
from sort_drift_sentinel.core.schemas import DriftReport, GarmentEvent, HealthStatus

from .mahalanobis import MahalanobisModel
from .mmd import mmd_rbf
from .psi import categorical_psi, population_stability_index


@dataclass
class Baseline:
    events: list[GarmentEvent]
    scalars: dict[str, np.ndarray]
    drift_matrix: np.ndarray
    grades: list[str]
    mahalanobis: MahalanobisModel
    p99_latency_ms: float


class DriftEngine:
    def __init__(self, window_size: int = 64, min_window: int = 24) -> None:
        self.window_size = window_size
        self.min_window = min_window
        self.baselines: dict[str, Baseline] = {}
        self.windows: dict[str, deque[GarmentEvent]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )

    def fit_baseline(self, deployment_id: str, events: list[GarmentEvent]) -> None:
        if len(events) < max(24, self.min_window):
            raise ValueError("baseline needs enough known-good events")
        scalar_rows = [scalar_features(e) for e in events]
        keys = scalar_rows[0].keys()
        scalars = {
            key: np.asarray([row[key] for row in scalar_rows], dtype=np.float64) for key in keys
        }
        matrix = np.vstack([drift_vector(e) for e in events])
        if len({len(row) for row in matrix}) != 1:
            raise ValueError("baseline drift vectors have inconsistent dimensions")
        mahal = MahalanobisModel().fit(matrix)
        self.baselines[deployment_id] = Baseline(
            events=events,
            scalars=scalars,
            drift_matrix=matrix,
            grades=[e.grade.grade for e in events],
            mahalanobis=mahal,
            p99_latency_ms=float(np.quantile(scalars["total_latency_ms"], 0.99)),
        )
        self.windows[deployment_id].clear()

    def update(self, event: GarmentEvent) -> DriftReport | None:
        deployment_id = event.deployment_id
        self.windows[deployment_id].append(event)
        if deployment_id not in self.baselines or len(self.windows[deployment_id]) < self.min_window:
            return None
        return self.evaluate(deployment_id)

    def evaluate(self, deployment_id: str) -> DriftReport:
        baseline = self.baselines[deployment_id]
        current = list(self.windows[deployment_id])
        rows = [scalar_features(e) for e in current]
        current_scalars = {
            key: np.asarray([row[key] for row in rows], dtype=np.float64)
            for key in baseline.scalars
        }

        metrics: dict[str, float] = {}
        for key in [
            "brightness",
            "blur_log",
            "nir_signal",
            "category_confidence",
            "grade_confidence",
            "ood_score",
            "total_latency_ms",
        ]:
            metrics[f"psi_{key}"] = population_stability_index(
                baseline.scalars[key], current_scalars[key]
            )

        metrics["psi_grade_distribution"] = categorical_psi(
            baseline.grades, [e.grade.grade for e in current]
        )

        current_matrix = np.vstack([drift_vector(e) for e in current])
        if current_matrix.shape[1] == baseline.drift_matrix.shape[1]:
            metrics["mmd"] = mmd_rbf(baseline.drift_matrix, current_matrix)
            mahal_scores = [baseline.mahalanobis.score(row) for row in current_matrix]
            metrics["mahalanobis_p95"] = float(np.quantile(mahal_scores, 0.95))
            metrics["mahalanobis_p95_ratio"] = metrics["mahalanobis_p95"] / max(baseline.mahalanobis.warning_, 1e-6)
        else:
            # Producer switched between fallback vector and true embedding dimension.
            metrics["mmd"] = 0.0
            metrics["mahalanobis_p95"] = 0.0
            metrics["mahalanobis_p95_ratio"] = 0.0

        current_latency_p99 = float(np.quantile(current_scalars["total_latency_ms"], 0.99))
        metrics["latency_p99_ms"] = current_latency_p99
        metrics["latency_p99_ratio"] = current_latency_p99 / max(
            baseline.p99_latency_ms, 1e-6
        )
        metrics["invalid_frame_rate"] = float(
            np.mean([0.0 if e.frame_quality.valid else 1.0 for e in current])
        )
        metrics["human_review_rate"] = float(
            np.mean([1.0 if e.route.requires_human else 0.0 for e in current])
        )

        input_status = self._status(
            warning=max(
                metrics["psi_brightness"],
                metrics["psi_blur_log"],
                metrics["psi_nir_signal"],
                metrics["invalid_frame_rate"] * 3.0,
            ),
            warn=0.35,
            crit=0.80,
        )
        model_status = self._status(
            warning=max(
                metrics["psi_category_confidence"],
                metrics["psi_grade_confidence"],
                metrics["psi_ood_score"],
                metrics["psi_grade_distribution"],
                metrics["mmd"] * 4.0,
                max(0.0, metrics["mahalanobis_p95_ratio"] - 1.0),
            ),
            warn=0.35,
            crit=0.80,
        )
        runtime_status = self._status(
            warning=max(0.0, metrics["latency_p99_ratio"] - 1.0),
            warn=0.25,
            crit=0.75,
        )
        business_status = self._status(
            warning=metrics["human_review_rate"], warn=0.15, crit=0.35
        )
        overall = self._worst(input_status, model_status, runtime_status, business_status)
        likely_cause, evidence, action = self._triage(metrics, input_status, model_status, runtime_status)

        return DriftReport(
            deployment_id=deployment_id,
            sample_count=len(current),
            input_health=input_status,
            model_health=model_status,
            runtime_health=runtime_status,
            business_health=business_status,
            overall=overall,
            metrics={k: round(float(v), 5) for k, v in metrics.items()},
            likely_cause=likely_cause,
            evidence=evidence,
            recommended_action=action,
        )

    @staticmethod
    def _status(warning: float, warn: float, crit: float) -> HealthStatus:
        if warning >= crit:
            return HealthStatus.CRITICAL
        if warning >= warn:
            return HealthStatus.WARNING
        return HealthStatus.HEALTHY

    @staticmethod
    def _worst(*statuses: HealthStatus) -> HealthStatus:
        rank = {
            HealthStatus.WARMING_UP: 0,
            HealthStatus.HEALTHY: 1,
            HealthStatus.WARNING: 2,
            HealthStatus.CRITICAL: 3,
        }
        return max(statuses, key=lambda s: rank[s])

    @staticmethod
    def _triage(
        m: dict[str, float],
        input_status: HealthStatus,
        model_status: HealthStatus,
        runtime_status: HealthStatus,
    ) -> tuple[str, list[str], str]:
        evidence: list[str] = []

        if runtime_status in {HealthStatus.WARNING, HealthStatus.CRITICAL}:
            evidence.append(f"p99 latency ratio vs baseline = {m['latency_p99_ratio']:.2f}x")
            return (
                "Edge runtime slowdown / saturation",
                evidence,
                "Check GPU utilization, queue depth, thermal throttling and the latest runtime/model build.",
            )

        lighting_signal = max(m["psi_brightness"], m["psi_blur_log"])
        if input_status in {HealthStatus.WARNING, HealthStatus.CRITICAL} and lighting_signal >= 0.35:
            evidence.extend(
                [
                    f"brightness PSI = {m['psi_brightness']:.2f}",
                    f"blur PSI = {m['psi_blur_log']:.2f}",
                    f"confidence PSI = {m['psi_category_confidence']:.2f}",
                ]
            )
            return (
                "Camera / lighting shift",
                evidence,
                "Inspect illumination, diffuser, exposure, lens cleanliness and camera focus before retraining.",
            )

        if input_status in {HealthStatus.WARNING, HealthStatus.CRITICAL} and m["psi_nir_signal"] >= 0.45:
            evidence.append(f"NIR signal PSI = {m['psi_nir_signal']:.2f}")
            return (
                "NIR sensor / calibration shift",
                evidence,
                "Check NIR calibration, saturation and sensor health; do not change the RGB model first.",
            )

        if model_status in {HealthStatus.WARNING, HealthStatus.CRITICAL}:
            evidence.extend(
                [
                    f"confidence PSI = {m['psi_category_confidence']:.2f}",
                    f"OOD PSI = {m['psi_ood_score']:.2f}",
                    f"MMD = {m['mmd']:.3f}",
                    f"grade-distribution PSI = {m['psi_grade_distribution']:.2f}",
                ]
            )
            return (
                "Product-mix / representation shift or model issue",
                evidence,
                "Sample uncertain/OOD garments for review, compare by model version, then validate labels before retraining or rollback.",
            )

        return (
            "No material drift detected",
            ["camera, model and runtime signals are within configured operating ranges"],
            "Continue monitoring. Do not retrain from a healthy window.",
        )
