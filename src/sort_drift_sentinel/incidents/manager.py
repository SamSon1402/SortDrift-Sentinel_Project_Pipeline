from __future__ import annotations

from collections import defaultdict

from sort_drift_sentinel.core.schemas import (
    DriftReport,
    GarmentEvent,
    HealthStatus,
    Incident,
    IncidentSeverity,
)


class IncidentManager:
    def __init__(self) -> None:
        self.open_incidents: dict[str, Incident] = {}
        self.history: list[Incident] = []
        self.consecutive_bad: dict[str, int] = defaultdict(int)
        self.consecutive_good: dict[str, int] = defaultdict(int)

    def process(self, report: DriftReport, last_event: GarmentEvent) -> Incident | None:
        deployment = report.deployment_id
        bad = report.overall in {HealthStatus.WARNING, HealthStatus.CRITICAL}
        if bad:
            self.consecutive_bad[deployment] += 1
            self.consecutive_good[deployment] = 0
        else:
            self.consecutive_good[deployment] += 1
            self.consecutive_bad[deployment] = 0

        # Debounce rolling-window noise: require two consecutive bad evaluations.
        if bad and self.consecutive_bad[deployment] >= 2:
            severity = (
                IncidentSeverity.CRITICAL
                if report.overall == HealthStatus.CRITICAL
                else IncidentSeverity.WARNING
            )
            existing = self.open_incidents.get(deployment)
            if existing:
                existing.severity = severity
                existing.evidence = report.evidence
                existing.recommended_action = report.recommended_action
                return None
            incident = Incident(
                deployment_id=deployment,
                severity=severity,
                title=f"{report.overall.value}: {report.likely_cause}",
                likely_cause=report.likely_cause,
                evidence=report.evidence,
                recommended_action=report.recommended_action,
                model_version=last_event.vision.model_version,
                camera_id=last_event.camera_id,
            )
            self.open_incidents[deployment] = incident
            self.history.append(incident)
            return incident

        if not bad and deployment in self.open_incidents and self.consecutive_good[deployment] >= 3:
            self.open_incidents[deployment].status = "RESOLVED"
            del self.open_incidents[deployment]
        return None
