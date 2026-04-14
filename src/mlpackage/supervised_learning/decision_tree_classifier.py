"""Entropy-based decision tree and random forest classifier (educational)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np

TreeNode = Union["LeafNode", "SplitNode"]


@dataclass(frozen=True)
class LeafNode:
    """Terminal tree node containing a predicted class."""

    class_label: int


@dataclass(frozen=True)
class SplitNode:
    """Internal tree node containing a split rule."""

    feature_index: int
    threshold: float
    left_child: TreeNode
    right_child: TreeNode


def _shannon_entropy(labels: np.ndarray) -> float:
    if labels.size == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / labels.size
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _gain_from_split(parent: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    if parent.size == 0:
        return 0.0
    left_weight = left.size / parent.size
    right_weight = right.size / parent.size
    return _shannon_entropy(parent) - (
        left_weight * _shannon_entropy(left) + right_weight * _shannon_entropy(right)
    )


def _majority_class(labels: np.ndarray) -> int:
    values, counts = np.unique(labels, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _optimal_binary_split(
    X: np.ndarray, y: np.ndarray
) -> Tuple[Optional[int], Optional[float]]:
    _, n_features = X.shape
    best_score = -1.0
    best_feature_index: Optional[int] = None
    best_threshold: Optional[float] = None

    for feature_index in range(n_features):
        candidates = np.unique(X[:, feature_index])

        for threshold in candidates:
            left_mask = X[:, feature_index] <= threshold
            right_mask = ~left_mask
            if not left_mask.any() or not right_mask.any():
                continue
            score = _gain_from_split(y, y[left_mask], y[right_mask])
            if score > best_score:
                best_score = score
                best_feature_index = feature_index
                best_threshold = float(threshold)

    return best_feature_index, best_threshold


class DecisionTreeClassifier:
    """Binary CART-style classifier with entropy / information-gain splits."""

    def __init__(self, max_depth: Optional[int] = None) -> None:
        self.max_depth = max_depth
        self._root: Optional[TreeNode] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        if X.size == 0 or y.size == 0:
            raise ValueError("Empty X or y provided.")
        if X.shape[0] != y.size:
            raise ValueError("Number of samples in X and y must match.")
        self._root = self._fit_node(X, y, depth=0)
        return self

    def _fit_node(self, X: np.ndarray, y: np.ndarray, depth: int) -> TreeNode:
        labels = np.unique(y)
        if labels.size == 1:
            return LeafNode(class_label=int(y[0]))

        if self.max_depth is not None and depth >= self.max_depth:
            return LeafNode(class_label=_majority_class(y))

        feature_index, threshold = _optimal_binary_split(X, y)
        if feature_index is None:
            return LeafNode(class_label=_majority_class(y))

        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask
        left_child = self._fit_node(X[left_mask], y[left_mask], depth + 1)
        right_child = self._fit_node(X[right_mask], y[right_mask], depth + 1)
        return SplitNode(
            feature_index=feature_index,
            threshold=threshold,
            left_child=left_child,
            right_child=right_child,
        )

    def _classify_row(self, row: np.ndarray, node: TreeNode) -> int:
        if isinstance(node, LeafNode):
            return node.class_label
        if row[node.feature_index] <= node.threshold:
            return self._classify_row(row, node.left_child)
        return self._classify_row(row, node.right_child)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._root is None:
            raise AttributeError("Model not fitted yet.")
        X = np.asarray(X)
        return np.array([self._classify_row(row, self._root) for row in X])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y).ravel()
        preds = self.predict(X)
        return float(np.mean(preds == y))


class RandomForestClassifier:
    """Bagging ensemble of DecisionTreeClassifier with optional column subsampling."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        max_features: Optional[str] = "sqrt",
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self._trees: List[Tuple[DecisionTreeClassifier, np.ndarray]] = []

    def _column_indices(self, n_features: int) -> np.ndarray:
        if self.max_features == "sqrt":
            num_selected = max(1, int(np.sqrt(n_features)))
            return np.random.choice(n_features, size=num_selected, replace=False)
        if self.max_features is None:
            return np.arange(n_features)
        raise ValueError("max_features must be 'sqrt' or None.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        n_samples, n_features = X.shape
        self._trees.clear()

        for _ in range(self.n_estimators):
            sample_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            feature_indices = self._column_indices(n_features)
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X[np.ix_(sample_indices, feature_indices)], y[sample_indices])
            self._trees.append((tree, feature_indices))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if not self._trees:
            raise AttributeError("Model not fitted yet.")
        tree_predictions = np.column_stack(
            [tree.predict(X[:, feature_indices]) for tree, feature_indices in self._trees]
        )
        majority_vote = np.empty(tree_predictions.shape[0], dtype=int)
        for row_index in range(tree_predictions.shape[0]):
            row_votes = tree_predictions[row_index].astype(int, copy=False)
            majority_vote[row_index] = int(np.bincount(row_votes).argmax())
        return majority_vote
