from __future__ import annotations

import json

from sort_drift_sentinel.service import SortDriftService
from sort_drift_sentinel.simulator.generator import GarmentEventGenerator
from sort_drift_sentinel.storage.local import JsonlStore


def run(deployment: str, mode: str, seed: int) -> None:
    generator = GarmentEventGenerator(seed=seed)
    service = SortDriftService(JsonlStore("runtime"), window_size=64, min_window=24)
    baseline = generator.generate(deployment, 96, mode="healthy", start_index=0)
    service.fit_baseline(deployment, baseline)
    current = GarmentEventGenerator(seed=seed + 100).generate(
        deployment, 64, mode=mode, start_index=1000
    )
    report = None
    for event in current:
        report, _ = service.ingest(event)
    print(f"\n{deployment} / {mode}")
    print(json.dumps(report.model_dump(mode="json"), indent=2))
    print("\nActive learning top 5:")
    for item in service.active_learning(deployment, 5):
        print(item.model_dump())


if __name__ == "__main__":
    run("retextil-line-01", "healthy", 7)
    run("texaid-line-02", "lighting_shift", 11)
    run("ecotextile-line-03", "critical_drift", 13)
