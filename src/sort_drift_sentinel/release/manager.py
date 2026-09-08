from __future__ import annotations

from sort_drift_sentinel.core.schemas import (
    ReleaseMetrics,
    ReleaseRecord,
    ReleaseState,
)

from .gate import ReleaseGate


class ReleaseManager:
    def __init__(self, gate: ReleaseGate | None = None) -> None:
        self.gate = gate or ReleaseGate()
        self.records: dict[str, ReleaseRecord] = {}

    def register(
        self,
        deployment_id: str,
        current_model: str,
        candidate_model: str,
        current_metrics: ReleaseMetrics,
        candidate_metrics: ReleaseMetrics,
    ) -> ReleaseRecord:
        result = self.gate.evaluate(current_metrics, candidate_metrics)
        record = ReleaseRecord(
            deployment_id=deployment_id,
            current_model=current_model,
            candidate_model=candidate_model,
            current_metrics=current_metrics,
            candidate_metrics=candidate_metrics,
            passed_gate=result.passed,
            gate_reasons=result.reasons,
            state=ReleaseState.REGISTERED if result.passed else ReleaseState.REJECTED,
        )
        self.records[record.release_id] = record
        return record

    def start_shadow(self, release_id: str) -> ReleaseRecord:
        record = self._get(release_id)
        if not record.passed_gate:
            raise ValueError("cannot shadow a release that failed the offline gate")
        if record.state not in {ReleaseState.REGISTERED, ReleaseState.SHADOW}:
            raise ValueError(f"cannot enter shadow from {record.state}")
        record.state = ReleaseState.SHADOW
        record.canary_percent = 0
        return record

    def start_canary(self, release_id: str, percent: int = 10) -> ReleaseRecord:
        record = self._get(release_id)
        if record.state != ReleaseState.SHADOW:
            raise ValueError("canary requires a successful shadow stage")
        if not 1 <= percent <= 50:
            raise ValueError("portfolio implementation limits canary to 1..50%")
        record.state = ReleaseState.CANARY
        record.canary_percent = percent
        return record

    def promote(self, release_id: str) -> ReleaseRecord:
        record = self._get(release_id)
        if record.state != ReleaseState.CANARY:
            raise ValueError("promotion requires canary")
        record.state = ReleaseState.PROMOTED
        record.canary_percent = 100
        return record

    def rollback(self, release_id: str) -> ReleaseRecord:
        record = self._get(release_id)
        if record.state not in {ReleaseState.SHADOW, ReleaseState.CANARY, ReleaseState.PROMOTED}:
            raise ValueError("rollback only applies after deployment has started")
        record.state = ReleaseState.ROLLED_BACK
        record.canary_percent = 0
        return record

    def _get(self, release_id: str) -> ReleaseRecord:
        if release_id not in self.records:
            raise KeyError(release_id)
        return self.records[release_id]
