from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

INGESTED = Counter("sortdrift_events_total", "Garment events ingested", ["deployment"])
INCIDENTS = Counter("sortdrift_incidents_total", "Incidents opened", ["deployment", "severity"])
LATENCY = Histogram(
    "sortdrift_garment_latency_ms",
    "Garment pipeline total latency",
    ["deployment"],
    buckets=(10, 20, 30, 40, 60, 80, 120, 200, 400),
)
DRIFT = Gauge("sortdrift_drift_metric", "Latest drift metric", ["deployment", "metric"])
HEALTH = Gauge(
    "sortdrift_health_state",
    "0 warming/unknown, 1 healthy, 2 warning, 3 critical",
    ["deployment", "layer"],
)
