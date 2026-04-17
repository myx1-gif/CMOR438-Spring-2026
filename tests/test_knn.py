import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from mlpackage.supervised_learning import KNeighborsClassifier


def test_knn_predict_simple():
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    y = np.array([0, 0, 1, 1])
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X, y)
    assert np.array_equal(clf.predict(X), y)


def test_knn_accuracy_and_score():
    X = np.array([[0], [1], [2], [3]], dtype=float)
    y = np.array([0, 0, 1, 1])
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X, y)
    assert clf.score(X, y) == 1.0
    assert clf.accuracy(X, y) == 1.0


def test_knn_confusion_matrix():
    X = np.array([[0], [1], [2], [3]], dtype=float)
    y = np.array([0, 0, 1, 1])
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X, y)
    cm = clf.confusion_matrix(X, y)
    assert cm.shape == (2, 2)
    assert cm.loc[0, 0] == 2
    assert cm.loc[1, 1] == 2


def test_knn_requires_fit():
    clf = KNeighborsClassifier()
    with pytest.raises(AttributeError):
        clf.predict(np.zeros((1, 2)))


def test_knn_empty_fit():
    clf = KNeighborsClassifier()
    with pytest.raises(ValueError):
        clf.fit(np.empty((0, 2)), np.array([]))


def test_knn_decision_boundary_runs_without_display(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X, y)
    clf.plot_decision_boundary(X, y)
    plt.close("all")


def test_knn_draw_decision_boundary_alias(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    y = np.array([0, 0, 1, 1])
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X, y)
    clf.draw_decision_boundary(X, y)
    plt.close("all")
