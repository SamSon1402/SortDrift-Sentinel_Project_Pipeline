from __future__ import annotations

import os

from sort_drift_sentinel.core.schemas import GarmentEvent, Incident, ReleaseRecord

from .base import Store


class SupabaseStore(Store):
    """Optional YC S20 persistence adapter using the official supabase-py client."""

    def __init__(self, url: str | None = None, key: str | None = None) -> None:
        url = url or os.getenv("SUPABASE_URL")
        key = key or os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY are required")
        try:
            from supabase import create_client
        except ImportError as exc:
            raise RuntimeError("install optional dependency: pip install -e '.[yc]'") from exc
        self.client = create_client(url, key)

    def save_event(self, event: GarmentEvent) -> None:
        self.client.table("sort_drift_events").insert(
            {
                "event_id": event.event_id,
                "deployment_id": event.deployment_id,
                "garment_id": event.garment_id,
                "model_version": event.vision.model_version,
                "payload": event.model_dump(mode="json"),
            }
        ).execute()

    def save_incident(self, incident: Incident) -> None:
        self.client.table("sort_drift_incidents").upsert(
            {
                "incident_id": incident.incident_id,
                "deployment_id": incident.deployment_id,
                "severity": incident.severity.value,
                "status": incident.status,
                "payload": incident.model_dump(mode="json"),
            },
            on_conflict="incident_id",
        ).execute()

    def save_release(self, release: ReleaseRecord) -> None:
        self.client.table("sort_drift_releases").upsert(
            {
                "release_id": release.release_id,
                "deployment_id": release.deployment_id,
                "state": release.state.value,
                "payload": release.model_dump(mode="json"),
            },
            on_conflict="release_id",
        ).execute()
