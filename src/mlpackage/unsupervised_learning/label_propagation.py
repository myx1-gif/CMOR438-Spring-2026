"""Graph-based semi-supervised label propagation with RBF similarity."""

from __future__ import annotations

from typing import Optional

import numpy as np


def _pairwise_squared_distances(X: np.ndarray) -> np.ndarray:
    """Compute the matrix of squared Euclidean distances between all rows."""
    row_norms = np.sum(X ** 2, axis=1)
    return row_norms[:, np.newaxis] + row_norms[np.newaxis, :] - 2.0 * (X @ X.T)


def _rbf_similarity(X: np.ndarray, bandwidth: float) -> np.ndarray:
    """Build an RBF (Gaussian) similarity matrix with zero self-affinity."""
    sq_dists = _pairwise_squared_distances(X)
    affinity = np.exp(-sq_dists / (2.0 * bandwidth ** 2))
    np.fill_diagonal(affinity, 0.0)
    return affinity


def _row_normalise(W: np.ndarray) -> np.ndarray:
    """Row-normalise a matrix so each row sums to 1 (or 0 for zero rows)."""
    totals = W.sum(axis=1, keepdims=True)
    totals = np.where(totals == 0, 1.0, totals)
    return W / totals


def _onehot_from_labels(y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Create a (n_samples, n_classes) one-hot matrix; unlabelled rows are zeros."""
    n = y.shape[0]
    n_classes = classes.shape[0]
    class_to_col = {int(c): idx for idx, c in enumerate(classes)}
    mat = np.zeros((n, n_classes), dtype=float)
    for i, label in enumerate(y):
        if int(label) in class_to_col:
            mat[i, class_to_col[int(label)]] = 1.0
    return mat


class GraphLabelPropagation:
    """
    Semi-supervised label propagation over an RBF similarity graph.

    Labelled samples are encoded with their class index (0, 1, 2, ...);
    unlabelled samples are marked with ``-1``.  The algorithm iteratively
    spreads soft label distributions from labelled to unlabelled nodes
    through a row-normalised transition matrix derived from an RBF kernel.

    Parameters
    ----------
    spread : float, default=0.9
        Propagation weight (0 < spread < 1).  Higher values let label
        information travel further per iteration.
    bandwidth : float, default=1.0
        RBF kernel width — controls how quickly similarity decays with
        distance.
    max_iterations : int, default=1000
        Upper bound on propagation sweeps.
    convergence_tol : float, default=1e-4
        Stop early when the Frobenius-norm change in the distribution
        matrix drops below this threshold.
    clamp_labelled : bool, default=True
        If ``True``, labelled nodes are reset to their ground-truth
        distribution after every iteration so they cannot be overwritten.

    Attributes
    ----------
    classes_ : np.ndarray | None
        Sorted array of unique observed classes (excluding -1).
    distribution_ : np.ndarray | None
        Soft label matrix of shape ``(n_samples, n_classes)`` after fitting.
    labels_ : np.ndarray | None
        Hard predictions (argmax over ``distribution_``).
    transition_ : np.ndarray | None
        Row-normalised affinity matrix used during propagation.
    """

    def __init__(
        self,
        spread: float = 0.9,
        bandwidth: float = 1.0,
        max_iterations: int = 1000,
        convergence_tol: float = 1e-4,
        clamp_labelled: bool = True,
    ) -> None:
        self.spread = spread
        self.bandwidth = bandwidth
        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol
        self.clamp_labelled = clamp_labelled

        self.classes_: Optional[np.ndarray] = None
        self.distribution_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.transition_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GraphLabelPropagation":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if X.shape[0] == 0:
            raise ValueError("Empty feature matrix provided.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        known_mask = y != -1
        self.classes_ = np.sort(np.unique(y[known_mask]))
        n_classes = self.classes_.shape[0]

        initial = _onehot_from_labels(y, self.classes_)

        W = _rbf_similarity(X, self.bandwidth)
        self.transition_ = _row_normalise(W)

        F = initial.copy()
        for _ in range(self.max_iterations):
            F_next = self.spread * (self.transition_ @ F) + (1.0 - self.spread) * initial

            if self.clamp_labelled:
                F_next[known_mask] = initial[known_mask]

            if np.linalg.norm(F_next - F) < self.convergence_tol:
                F = F_next
                break
            F = F_next

        self.distribution_ = F
        self.labels_ = self.classes_[np.argmax(F, axis=1)]
        return self

    def predict(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        if self.labels_ is None:
            raise AttributeError("Model has not been fitted yet.")
        return self.labels_
