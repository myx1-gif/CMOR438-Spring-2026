"""k-Nearest Neighbors classifier (educational, NumPy + optional pandas/matplotlib)."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _euclidean_rows(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance from one row `query` to each row in `reference`."""
    return np.sqrt(np.sum((reference - query) ** 2, axis=1))


def _majority_label(labels: np.ndarray) -> int:
    return int(np.bincount(labels.astype(int, copy=False)).argmax())


class KNeighborsClassifier:
    """
    k-Nearest Neighbors classifier using Euclidean distance and majority vote.

    Parameters
    ----------
    n_neighbors : int, default=3
        Number of neighbors to use for each prediction.
    """

    def __init__(self, n_neighbors: int = 3) -> None:
        self.n_neighbors = n_neighbors
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNeighborsClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        if X.size == 0 or y.size == 0:
            raise ValueError("Empty X or y provided.")
        if X.shape[0] != y.size:
            raise ValueError("Number of samples in X and y must match.")
        self._X_train = X
        self._y_train = y.astype(int, copy=False)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._X_train is None or self._y_train is None:
            raise AttributeError("Model not fitted yet.")
        X = np.asarray(X, dtype=float)
        n_refs = self._X_train.shape[0]
        k = min(self.n_neighbors, n_refs)
        out = np.empty(X.shape[0], dtype=int)
        for row_index in range(X.shape[0]):
            distances = _euclidean_rows(X[row_index], self._X_train)
            neighbor_order = np.argsort(distances)[:k]
            neighbor_labels = self._y_train[neighbor_order]
            out[row_index] = _majority_label(neighbor_labels)
        return out

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y).ravel()
        preds = self.predict(X)
        return float(np.mean(preds == y))

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Same as :meth:`score` (compatibility alias)."""
        return self.score(X, y)

    def confusion_matrix(self, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """Return a confusion matrix with rows = true labels, columns = predicted labels."""
        y_true = np.asarray(y).ravel()
        y_pred = self.predict(X)
        labels = np.unique(np.concatenate([y_true, y_pred]))
        frame = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
        for t, p in zip(y_true, y_pred):
            frame.loc[t, p] += 1
        return frame

    def plot_decision_boundary(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Plot a 2D decision surface (first two features only).

        Intended for small 2D classification demos.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] < 2:
            raise ValueError("X must have shape (n_samples, 2) or more (only first two columns used).")

        x1_min, x1_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
        x2_min, x2_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
        grid_x1, grid_x2 = np.meshgrid(
            np.arange(x1_min, x1_max, 0.1),
            np.arange(x2_min, x2_max, 0.1),
        )
        grid_points = np.c_[grid_x1.ravel(), grid_x2.ravel()]
        labels_grid = self.predict(grid_points).reshape(grid_x1.shape)

        plt.figure()
        plt.contourf(grid_x1, grid_x2, labels_grid, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=np.asarray(y), edgecolors="k", marker="o")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("KNN decision boundary")
        plt.show()

    def draw_decision_boundary(self, X: np.ndarray, y: np.ndarray) -> None:
        """Alias for :meth:`plot_decision_boundary` (same behavior)."""
        self.plot_decision_boundary(X, y)
