"""Variance-reduction decision tree and random forest regressor (educational)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np

RegressionNode = Union["RegressionLeaf", "RegressionSplit"]


@dataclass(frozen=True)
class RegressionLeaf:
    """Terminal node containing a prediction value."""

    value: float


@dataclass(frozen=True)
class RegressionSplit:
    """Internal node representing a binary split rule."""

    feature_index: int
    threshold: float
    left_child: RegressionNode
    right_child: RegressionNode


def _variance(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.var(values))


def _variance_drop(parent: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    if parent.size == 0:
        return 0.0
    left_weight = left.size / parent.size
    right_weight = right.size / parent.size
    return _variance(parent) - (
        left_weight * _variance(left) + right_weight * _variance(right)
    )


def _best_regression_split(
    X: np.ndarray, y: np.ndarray
) -> Tuple[Optional[int], Optional[float]]:
    _, n_features = X.shape
    best_gain = -1.0
    best_feature_index: Optional[int] = None
    best_threshold: Optional[float] = None

    for feature_index in range(n_features):
        for threshold in np.unique(X[:, feature_index]):
            left_mask = X[:, feature_index] <= threshold
            right_mask = ~left_mask
            if not left_mask.any() or not right_mask.any():
                continue
            gain = _variance_drop(y, y[left_mask], y[right_mask])
            if gain > best_gain:
                best_gain = gain
                best_feature_index = feature_index
                best_threshold = float(threshold)

    return best_feature_index, best_threshold


class DecisionTreeRegressor:
    """Simple binary decision tree regressor using variance reduction."""

    def __init__(self, max_depth: Optional[int] = None) -> None:
        self.max_depth = max_depth
        self._root: Optional[RegressionNode] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeRegressor":
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        if X.size == 0 or y.size == 0:
            raise ValueError("Empty X or y provided.")
        if X.shape[0] != y.size:
            raise ValueError("Number of samples in X and y must match.")
        self._root = self._grow(X, y, depth=0)
        return self

    def _grow(self, X: np.ndarray, y: np.ndarray, depth: int) -> RegressionNode:
        if np.unique(y).size == 1:
            return RegressionLeaf(value=float(np.mean(y)))
        if self.max_depth is not None and depth >= self.max_depth:
            return RegressionLeaf(value=float(np.mean(y)))

        feature_index, threshold = _best_regression_split(X, y)
        if feature_index is None:
            return RegressionLeaf(value=float(np.mean(y)))

        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask
        left_child = self._grow(X[left_mask], y[left_mask], depth + 1)
        right_child = self._grow(X[right_mask], y[right_mask], depth + 1)
        return RegressionSplit(
            feature_index=feature_index,
            threshold=threshold,
            left_child=left_child,
            right_child=right_child,
        )

    def _predict_row(self, row: np.ndarray, node: RegressionNode) -> float:
        if isinstance(node, RegressionLeaf):
            return node.value
        if row[node.feature_index] <= node.threshold:
            return self._predict_row(row, node.left_child)
        return self._predict_row(row, node.right_child)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._root is None:
            raise AttributeError("Model not fitted yet.")
        X = np.asarray(X)
        return np.array([self._predict_row(row, self._root) for row in X], dtype=float)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y).ravel().astype(float)
        pred = self.predict(X)
        ss_total = float(np.sum((y - y.mean()) ** 2))
        if ss_total == 0.0:
            return 1.0
        ss_res = float(np.sum((y - pred) ** 2))
        return 1.0 - (ss_res / ss_total)


class RandomForestRegressor:
    """Bagging ensemble of decision tree regressors."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        max_features: Optional[str] = "sqrt",
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self._trees: List[Tuple[DecisionTreeRegressor, np.ndarray]] = []

    def _feature_subset(self, n_features: int) -> np.ndarray:
        if self.max_features == "sqrt":
            num_selected = max(1, int(np.sqrt(n_features)))
            return np.random.choice(n_features, size=num_selected, replace=False)
        if self.max_features is None:
            return np.arange(n_features)
        raise ValueError("max_features must be 'sqrt' or None.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestRegressor":
        X = np.asarray(X)
        y = np.asarray(y).ravel().astype(float)
        if X.size == 0 or y.size == 0:
            raise ValueError("Empty X or y provided.")
        if X.shape[0] != y.size:
            raise ValueError("Number of samples in X and y must match.")

        n_samples, n_features = X.shape
        self._trees.clear()

        for _ in range(self.n_estimators):
            sample_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            feature_indices = self._feature_subset(n_features)
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X[np.ix_(sample_indices, feature_indices)], y[sample_indices])
            self._trees.append((tree, feature_indices))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if not self._trees:
            raise AttributeError("Model not fitted yet.")
        predictions = np.column_stack(
            [tree.predict(X[:, feature_indices]) for tree, feature_indices in self._trees]
        )
        return np.mean(predictions, axis=1)
