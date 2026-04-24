"""Density-based clustering (DBSCAN) using Euclidean neighbourhoods."""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np


def _neighbours_within_radius(
    data: np.ndarray, point_idx: int, radius: float
) -> np.ndarray:
    """Return indices of all points within `radius` of `data[point_idx]`."""
    diffs = data - data[point_idx]
    dists = np.sqrt(np.sum(diffs ** 2, axis=1))
    return np.nonzero(dists <= radius)[0]


class DBSCANClustering:
    """
    DBSCAN — Density-Based Spatial Clustering of Applications with Noise.

    Core points are those with at least `min_neighbours` points (including
    themselves) inside an `epsilon`-radius ball.  Non-core points reachable
    from a core point join the same cluster; all others are labelled noise (-1).

    Parameters
    ----------
    epsilon : float, default=0.5
        Neighbourhood radius.
    min_neighbours : int, default=5
        Minimum number of points required to form a dense region.

    Attributes
    ----------
    labels_ : np.ndarray | None
        Cluster assignment for every sample after fitting (-1 = noise).
    """

    def __init__(self, epsilon: float = 0.5, min_neighbours: int = 5) -> None:
        self.epsilon = epsilon
        self.min_neighbours = min_neighbours
        self.labels_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "DBSCANClustering":
        X = np.asarray(X, dtype=float)
        if X.size == 0:
            raise ValueError("Empty data provided.")
        n = X.shape[0]
        assignments = np.full(n, -1, dtype=int)
        explored = np.zeros(n, dtype=bool)
        current_cluster = 0

        for idx in range(n):
            if explored[idx]:
                continue
            explored[idx] = True
            nearby = _neighbours_within_radius(X, idx, self.epsilon)

            if nearby.size < self.min_neighbours:
                continue

            assignments[idx] = current_cluster
            expand_queue: deque[int] = deque(nearby.tolist())

            while expand_queue:
                candidate = expand_queue.popleft()
                if not explored[candidate]:
                    explored[candidate] = True
                    candidate_nearby = _neighbours_within_radius(
                        X, candidate, self.epsilon
                    )
                    if candidate_nearby.size >= self.min_neighbours:
                        expand_queue.extend(candidate_nearby.tolist())
                if assignments[candidate] == -1:
                    assignments[candidate] = current_cluster

            current_cluster += 1

        self.labels_ = assignments
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        assert self.labels_ is not None
        return self.labels_
