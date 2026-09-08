from __future__ import annotations

import numpy as np


def _safe_probs(counts: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    probs = counts.astype(np.float64)
    probs = probs / max(probs.sum(), 1.0)
    return np.clip(probs, eps, 1.0)


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 5,
) -> float:
    """PSI using bins learned from the reference distribution.

    PSI is a useful shift signal, not a proof that accuracy changed.
    """
    ref = np.asarray(reference, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)
    if ref.size < 5 or cur.size < 5:
        return 0.0

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(ref, quantiles)
    edges = np.unique(edges)
    if edges.size < 3:
        lo = min(float(ref.min()), float(cur.min())) - 1e-6
        hi = max(float(ref.max()), float(cur.max())) + 1e-6
        if hi <= lo:
            return 0.0
        edges = np.linspace(lo, hi, min(bins, 5) + 1)
    else:
        edges[0] = -np.inf
        edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    p = _safe_probs(ref_counts)
    q = _safe_probs(cur_counts)
    return float(np.sum((q - p) * np.log(q / p)))


def categorical_psi(reference: list[str], current: list[str]) -> float:
    labels = sorted(set(reference) | set(current))
    if not labels:
        return 0.0
    ref = np.asarray([reference.count(label) for label in labels], dtype=np.float64)
    cur = np.asarray([current.count(label) for label in labels], dtype=np.float64)
    p = _safe_probs(ref)
    q = _safe_probs(cur)
    return float(np.sum((q - p) * np.log(q / p)))
