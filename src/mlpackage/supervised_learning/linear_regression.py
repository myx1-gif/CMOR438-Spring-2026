"""Ordinary least squares linear regression via the normal equation (pseudoinverse)."""

from __future__ import annotations

from typing import Optional

import numpy as np


def _with_intercept_column(X: np.ndarray) -> np.ndarray:
    """Return design matrix with a leading column of ones (bias / intercept)."""
    X = np.asarray(X, dtype=float)
    n_samples = X.shape[0]
    ones = np.ones((n_samples, 1), dtype=float)
    return np.hstack([ones, X])


class LinearRegression:
    """
    Linear regression fit by solving the normal equation with a Moore–Penrose inverse:

        θ = (Xᵀ X)⁺ Xᵀ y

    where rows of X include a constant term for the intercept.

    Attributes
    ----------
    intercept : float | None
        Bias term after fitting.
    coef_ : np.ndarray | None
        Feature weights (shape ``(n_features,)``) after fitting.
    """

    def __init__(self) -> None:
        self.intercept: Optional[float] = None
        self.coef_: Optional[np.ndarray] = None
        self._is_fitted: bool = False

    @property
    def coefficients(self) -> Optional[np.ndarray]:
        """Alias for :attr:`coef_` (compatibility with older naming)."""
        return self.coef_

    @property
    def fitted(self) -> bool:
        """Whether ``fit`` has been successfully called on this instance."""
        return self._is_fitted

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """Estimate intercept and coefficients by ordinary least squares.

        Returns
        -------
        LinearRegression
            The fitted instance (``self``).
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        if X.size == 0 or y.size == 0:
            raise ValueError("Empty X or y provided to fit method.")
        if X.shape[0] != y.size:
            raise ValueError("Number of samples in X and y must match.")

        design = _with_intercept_column(X)
        gram = design.T @ design
        # moore-penrose inverse handles rank-deficient designs
        theta = np.linalg.pinv(gram) @ design.T @ y

        self.intercept = float(theta[0])
        self.coef_ = theta[1:].astype(float, copy=False)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return affine predictions ``X @ coef_ + intercept`` for each row."""
        if not self._is_fitted or self.coef_ is None or self.intercept is None:
            raise AttributeError("Model not fitted yet.")
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept

    def rmse(self, X: np.ndarray, y: np.ndarray) -> float:
        """Root mean squared error between ``y`` and ``predict(X)``."""
        y = np.asarray(y).ravel().astype(float)
        y_hat = self.predict(X)
        return float(np.sqrt(np.mean((y - y_hat) ** 2)))

    def R_squared(self, X: np.ndarray, y: np.ndarray) -> float:
        """Coefficient of determination :math:`R^2` relative to the mean of ``y``."""
        y = np.asarray(y).ravel().astype(float)
        y_hat = self.predict(X)
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot == 0.0:
            return 1.0
        ss_res = float(np.sum((y - y_hat) ** 2))
        return 1.0 - (ss_res / ss_tot)
