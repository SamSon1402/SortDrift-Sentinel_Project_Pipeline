from __future__ import annotations

import json
from pathlib import Path

from sort_drift_sentinel.core.schemas import GarmentEvent, Incident, ReleaseRecord

from .base import Store


class JsonlStore(Store):
    """Simple durable local sink. Cloud outages should not stop the line."""

    def __init__(self, root: str | Path = "runtime") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _append(self, name: str, payload: dict) -> None:
        path = self.root / f"{name}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str, separators=(",", ":")) + "\n")

    def save_event(self, event: GarmentEvent) -> None:
        self._append("events", event.model_dump(mode="json"))

    def save_incident(self, incident: Incident) -> None:
        self._append("incidents", incident.model_dump(mode="json"))

    def save_release(self, release: ReleaseRecord) -> None:
        self._append("releases", release.model_dump(mode="json"))
