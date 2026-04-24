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
from mlpackage.supervised_learning.logistic_regression import LogisticRegression
from mlpackage.supervised_learning.multilayer_perceptron import MultilayerPerceptron
from mlpackage.supervised_learning.perceptron import Perceptron

__all__ = [
    "DecisionTreeClassifier",
    "RandomForestClassifier",
    "DecisionTreeRegressor",
    "RandomForestRegressor",
    "KNeighborsClassifier",
    "LinearRegression",
    "LogisticRegression",
    "MultilayerPerceptron",
    "Perceptron",
]
