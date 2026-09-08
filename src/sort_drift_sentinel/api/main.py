from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from sort_drift_sentinel.core.schemas import GarmentEvent, ReleaseMetrics
from sort_drift_sentinel.simulator.generator import GarmentEventGenerator
from sort_drift_sentinel.storage.async_sink import AsyncStore
from sort_drift_sentinel.storage.composite import CompositeStore
from sort_drift_sentinel.storage.local import JsonlStore
from sort_drift_sentinel.storage.supabase import SupabaseStore
from sort_drift_sentinel.service import SortDriftService


def _build_store():
    stores = [JsonlStore(os.getenv("SORTDRIFT_LOCAL_DIR", "runtime"))]
    if os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_KEY"):
        try:
            stores.append(AsyncStore(SupabaseStore()))
        except Exception:
            # Local durability remains available even if the optional cloud adapter is misconfigured.
            pass
    return CompositeStore(stores)


app = FastAPI(title="SortDrift Sentinel", version="0.2.0")
service = SortDriftService(
    _build_store(),
    window_size=int(os.getenv("SORTDRIFT_WINDOW_SIZE", "64")),
    min_window=int(os.getenv("SORTDRIFT_MIN_WINDOW", "24")),
)


@app.get("/health/live")
def live():
    return {"status": "ok"}


@app.get("/health/ready")
def ready():
    return {"status": "ready", "baselines": list(service.engine.baselines)}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/dashboard")
def dashboard():
    here = Path(__file__).resolve()
    candidates = [Path.cwd() / "dashboard" / "index.html", here.parents[3] / "dashboard" / "index.html"]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise HTTPException(404, "dashboard file not found")
    return FileResponse(path)


@app.get("/v1/overview")
def overview():
    return service.overview()


@app.get("/v1/incidents")
def incidents():
    return [i.model_dump(mode="json") for i in service.incidents.history]


@app.get("/v1/active-learning/{deployment_id}")
def active_learning(deployment_id: str, k: int = 20):
    return [item.model_dump(mode="json") for item in service.active_learning(deployment_id, k)]


@app.post("/v1/baselines/{deployment_id}/demo")
def demo_baseline(deployment_id: str, n: int = 96):
    generator = GarmentEventGenerator(seed=11)
    events = generator.generate(deployment_id, n, mode="healthy", start_index=0)
    service.fit_baseline(deployment_id, events)
    return {"deployment_id": deployment_id, "baseline_samples": len(events)}


@app.post("/v1/events")
def ingest(event: GarmentEvent):
    report, incident = service.ingest(event)
    return {
        "accepted": True,
        "report": report.model_dump(mode="json") if report else None,
        "incident": incident.model_dump(mode="json") if incident else None,
    }


@app.post("/v1/demo/{deployment_id}/{mode}")
def demo(deployment_id: str, mode: str, n: int = 64):
    if deployment_id not in service.engine.baselines:
        generator = GarmentEventGenerator(seed=11)
        service.fit_baseline(
            deployment_id,
            generator.generate(deployment_id, 96, mode="healthy", start_index=0),
        )
    generator = GarmentEventGenerator(seed=42)
    last_report = None
    last_incident = None
    for event in generator.generate(deployment_id, n, mode=mode, start_index=1000):
        last_report, incident = service.ingest(event)
        if incident:
            last_incident = incident
    return {
        "report": last_report.model_dump(mode="json") if last_report else None,
        "incident": last_incident.model_dump(mode="json") if last_incident else None,
    }


@app.post("/v1/releases")
def register_release(payload: dict):
    try:
        return service.register_release(
            deployment_id=payload["deployment_id"],
            current_model=payload["current_model"],
            candidate_model=payload["candidate_model"],
            current_metrics=ReleaseMetrics.model_validate(payload["current_metrics"]),
            candidate_metrics=ReleaseMetrics.model_validate(payload["candidate_metrics"]),
        ).model_dump(mode="json")
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/v1/releases/{release_id}/shadow")
def release_shadow(release_id: str):
    try:
        record = service.releases.start_shadow(release_id)
        return service.save_release(record, "model_shadow_started").model_dump(mode="json")
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/v1/releases/{release_id}/canary")
def release_canary(release_id: str, percent: int = 10):
    try:
        record = service.releases.start_canary(release_id, percent)
        return service.save_release(record, "model_canary_started").model_dump(mode="json")
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/v1/releases/{release_id}/promote")
def release_promote(release_id: str):
    try:
        record = service.releases.promote(release_id)
        return service.save_release(record, "model_promoted").model_dump(mode="json")
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/v1/releases/{release_id}/rollback")
def release_rollback(release_id: str):
    try:
        record = service.releases.rollback(release_id)
        return service.save_release(record, "model_rolled_back").model_dump(mode="json")
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
