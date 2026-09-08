from __future__ import annotations

import logging

from sort_drift_sentinel.core.schemas import GarmentEvent, Incident, ReleaseRecord

from .base import Store

logger = logging.getLogger(__name__)


class CompositeStore(Store):
    """Local store can be mandatory while remote sinks are best-effort."""

    def __init__(self, stores: list[Store]) -> None:
        self.stores = stores

    def _fanout(self, method: str, value: object) -> None:
        for store in self.stores:
            try:
                getattr(store, method)(value)
            except Exception:
                logger.exception("optional store failed: %s", type(store).__name__)

    def save_event(self, event: GarmentEvent) -> None:
        self._fanout("save_event", event)

    def save_incident(self, incident: Incident) -> None:
        self._fanout("save_incident", incident)

    def save_release(self, release: ReleaseRecord) -> None:
        self._fanout("save_release", release)
