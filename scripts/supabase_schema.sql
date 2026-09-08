create table if not exists public.sort_drift_events (
    event_id text primary key,
    deployment_id text not null,
    garment_id text not null,
    model_version text not null,
    payload jsonb not null,
    created_at timestamptz not null default now()
);

create index if not exists sort_drift_events_deployment_created_idx
on public.sort_drift_events (deployment_id, created_at desc);

create table if not exists public.sort_drift_incidents (
    incident_id text primary key,
    deployment_id text not null,
    severity text not null,
    status text not null,
    payload jsonb not null,
    created_at timestamptz not null default now()
);

create table if not exists public.sort_drift_releases (
    release_id text primary key,
    deployment_id text not null,
    state text not null,
    payload jsonb not null,
    created_at timestamptz not null default now()
);

-- In a real deployment enable RLS and grant only the service role used by Sentinel.
