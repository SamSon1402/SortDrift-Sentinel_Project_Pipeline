from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass

from sort_drift_sentinel.core.schemas import GarmentEvent, Incident, ReleaseRecord

from .base import Store

logger = logging.getLogger(__name__)


@dataclass
class _Write:
    method: str
    value: object


class AsyncStore(Store):
    """Keep a remote/cloud sink out of the real-time ingest path.

    A bounded queue gives explicit backpressure semantics: if the remote sink is down
    for too long, the remote copy can be dropped while the local durable store remains
    the source of recovery data.
    """

    def __init__(self, inner: Store, max_queue: int = 5000) -> None:
        self.inner = inner
        self.queue: queue.Queue[_Write] = queue.Queue(maxsize=max_queue)
        self.worker = threading.Thread(target=self._run, name="sortdrift-remote-writer", daemon=True)
        self.worker.start()

    def _enqueue(self, method: str, value: object) -> None:
        try:
            self.queue.put_nowait(_Write(method, value))
        except queue.Full:
            logger.error("remote store queue full; local copy remains durable")

    def _run(self) -> None:
        while True:
            item = self.queue.get()
            try:
                getattr(self.inner, item.method)(item.value)
            except Exception:
                logger.exception("remote store write failed")
            finally:
                self.queue.task_done()

    def save_event(self, event: GarmentEvent) -> None:
        self._enqueue("save_event", event)

    def save_incident(self, incident: Incident) -> None:
        self._enqueue("save_incident", incident)

    def save_release(self, release: ReleaseRecord) -> None:
        self._enqueue("save_release", release)
