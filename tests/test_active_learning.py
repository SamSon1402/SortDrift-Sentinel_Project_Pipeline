from sort_drift_sentinel.active_learning.ranker import ActiveLearningRanker
from sort_drift_sentinel.simulator.generator import GarmentEventGenerator


def test_ranker_prefers_uncertain_ood_events():
    healthy = GarmentEventGenerator(seed=1).generate("x-line-01", 20, "healthy")
    drifted = GarmentEventGenerator(seed=2).generate("x-line-01", 20, "critical_drift", start_index=100)
    ranked = ActiveLearningRanker().rank(healthy + drifted, k=8)
    drift_ids = {e.event_id for e in drifted}
    assert sum(item.event_id in drift_ids for item in ranked) >= 5
