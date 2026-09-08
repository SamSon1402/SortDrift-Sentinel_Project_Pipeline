from __future__ import annotations

from collections import defaultdict, deque

from sort_drift_sentinel.active_learning.ranker import ActiveLearningRanker
from sort_drift_sentinel.core.schemas import DriftReport, GarmentEvent, HealthStatus, Incident, ReleaseRecord
from sort_drift_sentinel.drift.engine import DriftEngine
from sort_drift_sentinel.incidents.manager import IncidentManager
from sort_drift_sentinel.integrations.posthog import PostHogOps
from sort_drift_sentinel.release.manager import ReleaseManager
from sort_drift_sentinel.storage.base import Store
from sort_drift_sentinel.telemetry.metrics import DRIFT, HEALTH, INCIDENTS, INGESTED, LATENCY


class SortDriftService:
    def __init__(self, store: Store, window_size: int = 64, min_window: int = 24) -> None:
        self.store = store
        self.engine = DriftEngine(window_size=window_size, min_window=min_window)
        self.incidents = IncidentManager()
        self.ranker = ActiveLearningRanker()
        self.releases = ReleaseManager()
        self.posthog = PostHogOps()
        self.recent_events: dict[str, deque[GarmentEvent]] = defaultdict(lambda: deque(maxlen=500))
        self.latest_reports: dict[str, DriftReport] = {}

    def fit_baseline(self, deployment_id: str, events: list[GarmentEvent]) -> None:
        self.engine.fit_baseline(deployment_id, events)

    def ingest(self, event: GarmentEvent) -> tuple[DriftReport | None, Incident | None]:
        deployment = event.deployment_id
        self.store.save_event(event)
        self.recent_events[deployment].append(event)
        INGESTED.labels(deployment).inc()
        LATENCY.labels(deployment).observe(event.total_latency_ms)

        report = self.engine.update(event)
        if report is None:
            return None, None
        self.latest_reports[deployment] = report
        self._publish_report(report)

        incident = self.incidents.process(report, event)
        if incident:
            self.store.save_incident(incident)
            INCIDENTS.labels(deployment, incident.severity.value).inc()
            self.posthog.capture(
                deployment,
                "incident_opened",
                {
                    "severity": incident.severity.value,
                    "likely_cause": incident.likely_cause,
                    "model_version": incident.model_version,
                    "camera_id": incident.camera_id,
                },
            )
        return report, incident

    def active_learning(self, deployment_id: str, k: int = 20):
        return self.ranker.rank(list(self.recent_events[deployment_id]), k=k)

    def register_release(self, **kwargs) -> ReleaseRecord:
        record = self.releases.register(**kwargs)
        self.store.save_release(record)
        self.posthog.capture(
            record.deployment_id,
            "model_release_registered",
            {"candidate_model": record.candidate_model, "passed_gate": record.passed_gate},
        )
        return record

    def save_release(self, record: ReleaseRecord, event_name: str) -> ReleaseRecord:
        self.store.save_release(record)
        self.posthog.capture(
            record.deployment_id,
            event_name,
            {
                "candidate_model": record.candidate_model,
                "state": record.state.value,
                "canary_percent": record.canary_percent,
            },
        )
        return record

    def overview(self) -> list[dict]:
        deployments = set(self.engine.baselines) | set(self.latest_reports)
        rows = []
        for deployment in sorted(deployments):
            report = self.latest_reports.get(deployment)
            if report:
                rows.append(report.model_dump(mode="json"))
            else:
                rows.append(
                    {
                        "deployment_id": deployment,
                        "overall": HealthStatus.WARMING_UP.value,
                        "sample_count": len(self.engine.windows[deployment]),
                        "likely_cause": "collecting current window",
                        "recommended_action": "wait for enough current samples",
                        "metrics": {},
                    }
                )
        return rows

    @staticmethod
    def _publish_report(report: DriftReport) -> None:
        for name, value in report.metrics.items():
            DRIFT.labels(report.deployment_id, name).set(value)
        mapping = {
            HealthStatus.WARMING_UP: 0,
            HealthStatus.HEALTHY: 1,
            HealthStatus.WARNING: 2,
            HealthStatus.CRITICAL: 3,
        }
        HEALTH.labels(report.deployment_id, "input").set(mapping[report.input_health])
        HEALTH.labels(report.deployment_id, "model").set(mapping[report.model_health])
        HEALTH.labels(report.deployment_id, "runtime").set(mapping[report.runtime_health])
        HEALTH.labels(report.deployment_id, "business").set(mapping[report.business_health])
        HEALTH.labels(report.deployment_id, "overall").set(mapping[report.overall])
