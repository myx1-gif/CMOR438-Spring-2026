"""K-Means clustering via iterative centroid refinement."""

from __future__ import annotations

from typing import Optional

import numpy as np


def _closest_centroid(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Return the index of the nearest center for every row in *X*."""
    deltas = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(deltas ** 2, axis=2)
    return np.argmin(sq_dists, axis=1)


def _sum_of_squared_distances(
    X: np.ndarray, labels: np.ndarray, centers: np.ndarray, n_clusters: int
) -> float:
    """Total within-cluster sum of squared Euclidean distances."""
    total = 0.0
    for k in range(n_clusters):
        members = X[labels == k]
        total += float(np.sum((members - centers[k]) ** 2))
    return total


class KMeansClustering:
    """
    K-Means clustering.

    Repeatedly assigns each sample to its nearest centroid, then recomputes
    centroids as the mean of assigned samples.  Stops when centroids shift
    less than `convergence_tol` or `max_steps` iterations are reached.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters to form.
    max_steps : int, default=100
        Upper limit on iteration count.
    convergence_tol : float, default=1e-4
        If every centroid moves less than this between iterations, stop early.

    Attributes
    ----------
    centers_ : np.ndarray | None
        Coordinates of cluster centres after fitting.
    labels_ : np.ndarray | None
        Cluster index for each training sample.
    inertia_ : float | None
        Sum of squared distances of samples to their assigned centre.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        max_steps: int = 100,
        convergence_tol: float = 1e-4,
    ) -> None:
        self.n_clusters = n_clusters
        self.max_steps = max_steps
        self.convergence_tol = convergence_tol
        self.centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: Optional[float] = None

    def fit(self, X: np.ndarray, seed: int = 42) -> "KMeansClustering":
        X = np.asarray(X, dtype=float)
        if X.size == 0:
            raise ValueError("Empty dataset provided.")

        rng = np.random.RandomState(seed)
        chosen = rng.choice(X.shape[0], self.n_clusters, replace=False)
        self.centers_ = X[chosen].copy()

        for _ in range(self.max_steps):
            assignments = _closest_centroid(X, self.centers_)
            updated = np.array(
                [X[assignments == k].mean(axis=0) for k in range(self.n_clusters)]
            )
            shift = np.linalg.norm(updated - self.centers_)
            self.centers_ = updated
            if shift < self.convergence_tol:
                break

        self.labels_ = _closest_centroid(X, self.centers_)
        self.inertia_ = _sum_of_squared_distances(
            X, self.labels_, self.centers_, self.n_clusters
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.centers_ is None:
            raise AttributeError("Model has not been fitted yet.")
        return _closest_centroid(np.asarray(X, dtype=float), self.centers_)

    def score(self, X: np.ndarray) -> float:
        if self.centers_ is None:
            raise AttributeError("Model has not been fitted yet.")
        X = np.asarray(X, dtype=float)
        labels = _closest_centroid(X, self.centers_)
        return -_sum_of_squared_distances(
            X, labels, self.centers_, self.n_clusters
        )
