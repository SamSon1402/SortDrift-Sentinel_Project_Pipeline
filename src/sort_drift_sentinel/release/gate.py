from __future__ import annotations

from dataclasses import dataclass

from sort_drift_sentinel.core.schemas import ReleaseMetrics


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reasons: list[str]


class ReleaseGate:
    def __init__(
        self,
        max_p99_ms: float = 80.0,
        min_route_agreement: float = 0.95,
        max_quality_regression: float = 0.005,
    ) -> None:
        self.max_p99_ms = max_p99_ms
        self.min_route_agreement = min_route_agreement
        self.max_quality_regression = max_quality_regression

    def evaluate(self, current: ReleaseMetrics, candidate: ReleaseMetrics) -> GateResult:
        reasons: list[str] = []
        if candidate.defect_recall < current.defect_recall - self.max_quality_regression:
            reasons.append("defect recall regressed beyond allowed budget")
        if candidate.grade_macro_f1 < current.grade_macro_f1 - self.max_quality_regression:
            reasons.append("grade macro-F1 regressed beyond allowed budget")
        if candidate.route_agreement < self.min_route_agreement:
            reasons.append("route agreement is below the production floor")
        if candidate.route_agreement < current.route_agreement - self.max_quality_regression:
            reasons.append("route agreement regressed beyond allowed budget")
        if candidate.p99_latency_ms > self.max_p99_ms:
            reasons.append("candidate p99 latency exceeds absolute budget")
        if candidate.p99_latency_ms > current.p99_latency_ms * 1.20:
            reasons.append("candidate p99 latency is >20% slower than current")
        if candidate.manual_review_rate > current.manual_review_rate + 0.02:
            reasons.append("candidate increases manual review rate by >2pp")
        return GateResult(passed=not reasons, reasons=reasons or ["all release gates passed"])
