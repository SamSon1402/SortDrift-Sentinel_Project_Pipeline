# Architecture

```text
GarmentGrader edge runtime
        |
        | GarmentEvent
        v
+-------------------------------+
| SortDrift ingest              |
| schema + deployment metadata  |
+---------------+---------------+
                |
     +----------+----------+----------------+
     |                     |                |
     v                     v                v
Input health          Model health     Runtime health
brightness            confidence       p99 latency
blur/focus            OOD              inference ms
NIR signal            PSI              dropped frames*
invalid frames        MMD              restarts*
                      Mahalanobis
     |                     |                |
     +----------+----------+----------------+
                v
          Triage layer
                |
                v
        Incident / action
                |
                +--> active-learning queue
                |       |
                |       v
                |    human labels
                |       |
                |       v
                |    retraining
                |
                +--> release gate
                        |
                  shadow -> canary
                        |
                 promote / rollback
```

`*` Hardware-agent metrics are modeled as an extension point. The portfolio repo does not pretend it is talking to a real Jetson fleet.

## Storage boundary

- Local JSONL: durable fallback / demo.
- Supabase Postgres: optional event/incident/release state.
- Prometheus: time-series metrics.
- Selected failure images would normally go to object storage; this repo stores only event metadata.

## Important operating rule

Drift signal != confirmed accuracy drop.

Sentinel opens an investigation and selects representative samples. It does not silently retrain from one PSI/MMD alarm.
