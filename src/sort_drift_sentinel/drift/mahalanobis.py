from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf


class MahalanobisModel:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.precision_: np.ndarray | None = None
        self.warning_: float = 0.0
        self.critical_: float = 0.0

    def fit(self, reference: np.ndarray) -> "MahalanobisModel":
        x = np.asarray(reference, dtype=np.float64)
        if x.ndim != 2 or len(x) < 8:
            raise ValueError("need >=8 2D reference samples")
        model = LedoitWolf().fit(x)
        self.mean_ = model.location_
        self.precision_ = model.precision_
        train_scores = np.asarray([self.score(row) for row in x])
        self.warning_ = float(np.quantile(train_scores, 0.975))
        self.critical_ = float(np.quantile(train_scores, 0.995))
        return self

    def score(self, vector: np.ndarray) -> float:
        if self.mean_ is None or self.precision_ is None:
            raise RuntimeError("MahalanobisModel is not fitted")
        delta = np.asarray(vector, dtype=np.float64) - self.mean_
        return float(np.sqrt(max(0.0, delta @ self.precision_ @ delta.T)))
