import numpy as np

from sort_drift_sentinel.drift.psi import population_stability_index


def test_psi_is_small_for_similar_and_large_for_shift():
    rng = np.random.default_rng(4)
    ref = rng.normal(0, 1, 500)
    same = rng.normal(0, 1, 500)
    shifted = rng.normal(2.0, 1, 500)
    assert population_stability_index(ref, same) < population_stability_index(ref, shifted)
    assert population_stability_index(ref, shifted) > 0.2
