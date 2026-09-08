from __future__ import annotations

import numpy as np
from sklearn.metrics import pairwise_distances


def _rbf(x: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
    d2 = pairwise_distances(x, y, metric="sqeuclidean")
    return np.exp(-gamma * d2)


def _median_gamma(reference: np.ndarray, current: np.ndarray) -> float:
    sample = np.vstack([reference, current])
    if len(sample) > 256:
        idx = np.linspace(0, len(sample) - 1, 256).astype(int)
        sample = sample[idx]
    distances = pairwise_distances(sample, metric="euclidean")
    positive = distances[distances > 0]
    median = float(np.median(positive)) if positive.size else 1.0
    sigma2 = max(median * median, 1e-9)
    return 1.0 / (2.0 * sigma2)


def mmd_rbf(reference: np.ndarray, current: np.ndarray, gamma: float | None = None) -> float:
    """Biased MMD^2 estimator. Stable enough for a rolling operational signal."""
    x = np.asarray(reference, dtype=np.float64)
    y = np.asarray(current, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("reference/current must be 2D with the same feature dimension")
    if len(x) < 3 or len(y) < 3:
        return 0.0
    g = gamma or _median_gamma(x, y)
    kxx = _rbf(x, x, g)
    kyy = _rbf(y, y, g)
    kxy = _rbf(x, y, g)
    return float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())
