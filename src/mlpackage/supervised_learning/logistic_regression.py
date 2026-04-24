"""Binary logistic regression trained with batch gradient descent."""

from __future__ import annotations

from typing import Optional

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid: σ(z) = 1 / (1 + exp(−z))."""
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


class LogisticRegression:
    """
    Logistic regression classifier using batch gradient descent on binary
    cross-entropy loss.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size for gradient updates.
    n_iterations : int, default=1000
        Number of gradient descent iterations.

    Attributes
    ----------
    weights : np.ndarray | None
        Feature weight vector (shape ``(n_features,)``).
    bias : float | None
        Intercept / bias term.
    """

    def __init__(
        self, learning_rate: float = 0.01, n_iterations: int = 1000
    ) -> None:
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[float] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel().astype(float)
        if X.size == 0 or y.size == 0:
            raise ValueError("Empty X or y provided to fit method.")
        if X.shape[0] != y.size:
            raise ValueError("Number of samples in X and y must match.")

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0

        for _ in range(self.n_iterations):
            linear_term = X @ self.weights + self.bias
            predictions = _sigmoid(linear_term)

            gradient_w = (1.0 / n_samples) * (X.T @ (predictions - y))
            gradient_b = (1.0 / n_samples) * np.sum(predictions - y)

            self.weights -= self.learning_rate * gradient_w
            self.bias -= self.learning_rate * gradient_b

        return self

    def predict_probability(self, X: np.ndarray) -> np.ndarray:
        """Return P(y=1 | X) for each sample."""
        if self.weights is None or self.bias is None:
            raise AttributeError("Model not fitted yet.")
        X = np.asarray(X, dtype=float)
        return _sigmoid(X @ self.weights + self.bias)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_probability(X)
        return (probabilities >= 0.5).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y).ravel()
        return float(np.mean(self.predict(X) == y))
