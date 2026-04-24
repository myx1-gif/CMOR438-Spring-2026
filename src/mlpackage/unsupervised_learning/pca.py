"""Principal Component Analysis via eigen-decomposition of the covariance matrix."""

from __future__ import annotations

from typing import Optional

import numpy as np


class PrincipalComponentAnalysis:
    """
    Dimensionality reduction by projecting data onto its top eigenvectors.

    The covariance matrix of the centred data is decomposed with
    ``numpy.linalg.eigh``; the eigenvectors corresponding to the largest
    eigenvalues form the new basis.

    Parameters
    ----------
    n_components : int | None, default=None
        How many principal axes to retain.  ``None`` keeps all of them.

    Attributes
    ----------
    axes_ : np.ndarray | None
        Principal axes, shape ``(n_components, n_features)``.
    eigenvalues_ : np.ndarray | None
        Variance captured by each retained axis.
    variance_ratio_ : np.ndarray | None
        Fraction of total variance captured by each axis.
    feature_mean_ : np.ndarray | None
        Per-feature mean computed during ``fit``.
    """

    def __init__(self, n_components: Optional[int] = None) -> None:
        self.n_components = n_components
        self.axes_: Optional[np.ndarray] = None
        self.eigenvalues_: Optional[np.ndarray] = None
        self.variance_ratio_: Optional[np.ndarray] = None
        self.feature_mean_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "PrincipalComponentAnalysis":
        X = np.asarray(X, dtype=float)
        if X.size == 0:
            raise ValueError("Empty data provided.")

        self.feature_mean_ = X.mean(axis=0)
        centred = X - self.feature_mean_

        cov = np.cov(centred, rowvar=False)
        if cov.ndim == 0:
            cov = cov.reshape(1, 1)

        vals, vecs = np.linalg.eigh(cov)

        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]

        k = self.n_components if self.n_components is not None else vals.shape[0]
        self.eigenvalues_ = vals[:k]
        self.axes_ = vecs[:, :k].T

        total_var = vals.sum()
        self.variance_ratio_ = (
            self.eigenvalues_ / total_var if total_var > 0 else np.zeros(k)
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.axes_ is None:
            raise AttributeError("Model has not been fitted yet.")
        X = np.asarray(X, dtype=float)
        return (X - self.feature_mean_) @ self.axes_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)
