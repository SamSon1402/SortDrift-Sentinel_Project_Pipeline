# SortDrift Sentinel

SortDrift Sentinel watches a garment-sorting CV system after it is deployed.

It consumes the same `GarmentEvent` produced by GarmentGrader and answers four practical questions:

1. Is the camera/input changing?
2. Is the model drifting or seeing out-of-distribution garments?
3. Is the edge runtime getting slower or unstable?
4. Which samples should a human review, and is a candidate model safe to release?

The goal is not to draw a pretty drift chart. The goal is to turn model + camera + runtime signals into an incident an engineer can act on.

## Flow

```text
GARMENTGRADER
   |
   | GarmentEvent
   v
SORTDRIFT INGEST
   |
   +--> input health: brightness / blur / NIR signal
   +--> model health: confidence / OOD / output shift / MMD
   +--> runtime health: p95/p99 latency / restarts / dropped frames
   |
   v
ROOT-CAUSE TRIAGE
   |
   +--> lighting/camera issue
   +--> data/product-mix shift
   +--> model/release issue
   +--> runtime saturation
   |
   v
INCIDENT + ACTIVE-LEARNING QUEUE
   |
   v
HUMAN REVIEW -> RETRAIN -> SHADOW -> CANARY -> PROMOTE/ROLLBACK
```

## What is real in this repo

- Python + FastAPI service.
- Compatible parser for the GarmentGrader event contract.
- Rolling deployment windows.
- PSI for univariate distribution shift.
- MMD with an RBF kernel for multivariate drift.
- Mahalanobis/OOD scoring with Ledoit-Wolf covariance shrinkage.
- Root-cause triage using camera, model and runtime evidence together.
- Active-learning ranking using uncertainty + OOD + business risk + diversity.
- Release gate for defect recall, grade macro-F1, route agreement, manual-review rate and p99 latency.
- Shadow/canary/promote/rollback state machine.
- Prometheus metrics.
- Local JSONL storage.
- Optional Supabase persistence.
- Optional PostHog event analytics for incident/release events.
- Small operations dashboard.
- Simulation for healthy, lighting-shift and critical-drift deployments.
- Unit tests.

This is intentionally about 80% of a production implementation. Hardware-specific camera agents, PLC integration, a real annotation UI and customer authentication are left as explicit boundaries rather than faked.

## YC products used

### [Supabase — YC S20](https://www.ycombinator.com/companies/supabase)

Supabase is used as an optional Postgres-backed store for events, incidents and releases. It is **not** in the edge hot path. Remote Supabase writes run through a bounded background queue, so Supabase is not on the ingest hot path. If it is unavailable, the sorting line keeps running and the local JSONL store remains available.

### [PostHog — YC W20](https://www.ycombinator.com/companies/posthog)

PostHog is an optional event-analytics sink for operational events such as `incident_opened`, `model_canary_started` and `model_promoted`. It is useful for understanding how often deployments enter warning/critical states. It is not the source of truth for model quality.

## YC companies that influenced the product design

These are architecture/product references, not dependencies and not copied branding:

- **[Control Seat — YC S26](https://www.ycombinator.com/companies/control-seat):** one operational view combining control, monitoring, history and machine health.
- **[Cerrion — YC S22](https://www.ycombinator.com/companies/cerrion):** turn video/production signals into understandable incidents, not raw charts only.
- **[Optifye.ai — YC W25](https://www.ycombinator.com/companies/optifye-ai):** convert factory-camera output into live operational metrics.
- **[Grip — YC S26](https://www.ycombinator.com/companies/grip):** learn from real operating outcomes and failures, closing the loop between prediction and what actually happened.

## Why this design

A production CV service can be green while the model is wrong. A model can also look wrong when the real problem is a dirty lens, changed lighting or a bad NIR calibration.

So Sentinel separates:

```text
CAMERA HEALTH != MODEL HEALTH != RUNTIME HEALTH != BUSINESS HEALTH
```

A drift alarm is **not** an automatic retraining trigger. It creates evidence, an incident and a review queue. Labels are required before a model release is promoted.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]

python scripts/run_simulation.py
uvicorn sort_drift_sentinel.api.main:app --reload --port 8010
```

Open:

```text
http://localhost:8010/dashboard
http://localhost:8010/docs
http://localhost:8010/metrics
```

Run tests:

```bash
pytest -q
```

## Connect to GarmentGrader

GarmentGrader can write JSONL events. Replay them into Sentinel with:

```bash
python scripts/replay_from_garment_grader.py /path/to/garment_events.jsonl
```

The parser accepts the first project's event contract and also reads optional metadata fields when present:

```json
{
  "metadata": {
    "deployment_id": "texaid-line-02",
    "camera_id": "rgb-02",
    "taxonomy_version": "texaid-v42",
    "embedding": [0.12, -0.31, 0.44, 0.05]
  }
}
```

If no embedding vector is present, Sentinel falls back to a compact operational feature vector for MMD/diversity calculations.

## Example incident

```text
TEXAID / LINE 02                         WARNING

Input health       WARNING
Model health       WARNING
Runtime health     HEALTHY

Likely cause:
Camera/lighting shift on rgb-02.

Evidence:
brightness PSI = 0.31
blur PSI       = 0.08
confidence PSI = 0.27
latency p99    = stable

Action:
Inspect illumination/diffuser/camera exposure before touching the model.
```

That is the point of the project: explain what changed and what to do next.
