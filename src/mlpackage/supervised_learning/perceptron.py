"""Single-layer perceptron for binary classification (step-function activation)."""

from __future__ import annotations

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _step_activation(values: np.ndarray) -> np.ndarray:
    """Threshold activation: 1 where input >= 0, else 0."""
    return np.where(values >= 0, 1, 0)


class Perceptron:
    """
    Binary perceptron classifier trained with the classic online update rule.

    Parameters
    ----------
    lr : float, default=0.01
        Learning rate controlling the magnitude of weight corrections.
    max_iter : int, default=1000
        Number of full passes over the training data.

    Attributes
    ----------
    coef_ : np.ndarray | None
        Feature weight vector after fitting.
    intercept_ : float | None
        Bias term after fitting.
    training_errors : list[float]
        Mean squared error recorded at the end of each epoch.
    """

    def __init__(self, lr: float = 0.01, max_iter: int = 1000) -> None:
        self.lr = lr
        self.max_iter = max_iter
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self.training_errors: List[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        if X.size == 0 or y.size == 0:
            raise ValueError("Empty X or y provided.")
        if X.shape[0] != y.size:
            raise ValueError("Number of samples in X and y must match.")

        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features, dtype=float)
        self.intercept_ = 0.0
        self.training_errors = []

        for _ in range(self.max_iter):
            for i in range(n_samples):
                raw = X[i] @ self.coef_ + self.intercept_
                predicted = 1 if raw >= 0 else 0
                correction = self.lr * (y[i] - predicted)
                self.coef_ += correction * X[i]
                self.intercept_ += correction

            epoch_preds = self.predict(X)
            mse = float(np.mean((y - epoch_preds) ** 2))
            self.training_errors.append(mse)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise AttributeError("Model not fitted yet.")
        X = np.asarray(X, dtype=float)
        return _step_activation(X @ self.coef_ + self.intercept_)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y).ravel()
        return float(np.mean(self.predict(X) == y))

    def confusion_matrix(self, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """Return a confusion matrix (rows = actual, columns = predicted)."""
        y_true = np.asarray(y).ravel()
        y_pred = self.predict(X)
        return pd.crosstab(y_true, y_pred, rownames=["Actual"], colnames=["Predicted"])

    def plot_training_loss(self) -> None:
        """Plot MSE loss curve over training epochs."""
        if not self.training_errors:
            return
        plt.figure()
        plt.plot(range(len(self.training_errors)), self.training_errors, marker="o", markersize=2)
        plt.title("Perceptron training loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.grid(True)
        plt.show()
