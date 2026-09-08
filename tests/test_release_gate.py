import pytest

from sort_drift_sentinel.core.schemas import ReleaseMetrics, ReleaseState
from sort_drift_sentinel.release.manager import ReleaseManager


def metrics(recall=0.92, f1=0.93, route=0.97, p99=45, review=0.07):
    return ReleaseMetrics(
        defect_recall=recall,
        grade_macro_f1=f1,
        route_agreement=route,
        p99_latency_ms=p99,
        manual_review_rate=review,
    )


def test_good_candidate_can_move_shadow_canary_promote():
    manager = ReleaseManager()
    release = manager.register("d", "0.8", "0.9", metrics(), metrics(0.94, 0.94, 0.98, 49, 0.06))
    assert release.passed_gate
    assert manager.start_shadow(release.release_id).state == ReleaseState.SHADOW
    assert manager.start_canary(release.release_id, 10).state == ReleaseState.CANARY
    assert manager.promote(release.release_id).state == ReleaseState.PROMOTED


def test_bad_latency_is_rejected():
    manager = ReleaseManager()
    release = manager.register("d", "0.8", "slow", metrics(), metrics(p99=120))
    assert not release.passed_gate
    assert release.state == ReleaseState.REJECTED
    with pytest.raises(ValueError):
        manager.start_shadow(release.release_id)
