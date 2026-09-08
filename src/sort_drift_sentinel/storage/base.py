from __future__ import annotations

from abc import ABC, abstractmethod

from sort_drift_sentinel.core.schemas import GarmentEvent, Incident, ReleaseRecord


class Store(ABC):
    @abstractmethod
    def save_event(self, event: GarmentEvent) -> None: ...

    @abstractmethod
    def save_incident(self, incident: Incident) -> None: ...

    @abstractmethod
    def save_release(self, release: ReleaseRecord) -> None: ...
