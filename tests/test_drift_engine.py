from sort_drift_sentinel.core.schemas import HealthStatus
from sort_drift_sentinel.drift.engine import DriftEngine
from sort_drift_sentinel.simulator.generator import GarmentEventGenerator


def _report(mode: str):
    gen = GarmentEventGenerator(seed=10)
    engine = DriftEngine(window_size=64, min_window=24)
    deployment = "texaid-line-02"
    engine.fit_baseline(deployment, gen.generate(deployment, 96, "healthy"))
    report = None
    for event in GarmentEventGenerator(seed=99).generate(deployment, 64, mode, start_index=1000):
        report = engine.update(event)
    assert report is not None
    return report


def test_healthy_is_not_critical():
    assert _report("healthy").overall != HealthStatus.CRITICAL


def test_lighting_shift_points_to_camera():
    report = _report("lighting_shift")
    assert report.input_health in {HealthStatus.WARNING, HealthStatus.CRITICAL}
    assert "Camera / lighting" in report.likely_cause


def test_runtime_slowdown_points_to_runtime():
    report = _report("runtime_slowdown")
    assert report.runtime_health in {HealthStatus.WARNING, HealthStatus.CRITICAL}
    assert "runtime" in report.likely_cause.lower()
