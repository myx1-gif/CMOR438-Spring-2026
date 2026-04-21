"""Supervised learning algorithms (classification and regression)."""

from mlpackage.supervised_learning.decision_tree_classifier import (
    DecisionTreeClassifier,
    RandomForestClassifier,
)
from mlpackage.supervised_learning.decision_tree_regressor import (
    DecisionTreeRegressor,
    RandomForestRegressor,
)
from mlpackage.supervised_learning.knn import KNeighborsClassifier
from mlpackage.supervised_learning.linear_regression import LinearRegression

__all__ = [
    "DecisionTreeClassifier",
    "RandomForestClassifier",
    "DecisionTreeRegressor",
    "RandomForestRegressor",
    "KNeighborsClassifier",
    "LinearRegression",
]
