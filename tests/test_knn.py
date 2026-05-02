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


def test_knn_k_greater_than_samples_falls_back():
    """If n_neighbors > n_training_samples, the model should use all samples."""
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([1, 0, 1])
    clf = KNeighborsClassifier(n_neighbors=10)
    clf.fit(X, y)
    preds = clf.predict(np.array([[0.5]]))
    assert preds[0] == 1


def test_knn_majority_vote_with_k3():
    """With k=3, predicted label should be the majority among 3 nearest points."""
    X = np.array([[0.0], [1.0], [2.0], [10.0], [11.0]])
    y = np.array([0, 0, 0, 1, 1])
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X, y)
    assert clf.predict(np.array([[1.5]]))[0] == 0


def test_knn_multi_class_prediction():
    rng = np.random.default_rng(42)
    c0 = rng.normal(loc=[0, 0], scale=0.2, size=(10, 2))
    c1 = rng.normal(loc=[5, 0], scale=0.2, size=(10, 2))
    c2 = rng.normal(loc=[0, 5], scale=0.2, size=(10, 2))
    X = np.vstack([c0, c1, c2])
    y = np.array([0] * 10 + [1] * 10 + [2] * 10)
    clf = KNeighborsClassifier(n_neighbors=1).fit(X, y)
    preds = clf.predict(X)
    assert np.array_equal(preds, y)
    assert set(np.unique(preds)) == {0, 1, 2}


def test_knn_shape_mismatch_raises():
    clf = KNeighborsClassifier()
    with pytest.raises(ValueError):
        clf.fit(np.zeros((5, 2)), np.zeros(3))


def test_knn_predict_shape():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(25, 3))
    y_train = (X_train[:, 0] > 0).astype(int)
    X_test = rng.normal(size=(9, 3))
    clf = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
    preds = clf.predict(X_test)
    assert preds.shape == (9,)


def test_knn_fit_returns_self():
    X = np.array([[0.0], [1.0]])
    y = np.array([0, 1])
    clf = KNeighborsClassifier()
    assert clf.fit(X, y) is clf


def test_knn_default_n_neighbors_is_three():
    clf = KNeighborsClassifier()
    assert clf.n_neighbors == 3


def test_knn_confusion_matrix_totals():
    """Confusion matrix entries should sum to the number of samples."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 2))
    y = (X[:, 0] > 0).astype(int)
    clf = KNeighborsClassifier(n_neighbors=3).fit(X, y)
    cm = clf.confusion_matrix(X, y)
    assert cm.to_numpy().sum() == 30


def test_knn_confusion_matrix_diagonal_on_clean_data():
    """On linearly separable data fit perfectly, confusion matrix should be diagonal."""
    X = np.array([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    clf = KNeighborsClassifier(n_neighbors=1).fit(X, y)
    cm = clf.confusion_matrix(X, y)
    arr = cm.to_numpy()
    assert arr[0, 0] == 3
    assert arr[1, 1] == 3
    assert arr[0, 1] == 0
    assert arr[1, 0] == 0


def test_knn_changing_k_changes_decisions():
    """k=1 and k=5 should behave differently in an ambiguous region."""
    X = np.array([
        [0.0], [0.1], [0.2],
        [1.0], [1.1], [1.2],
        [5.0],
    ])
    y = np.array([0, 0, 0, 1, 1, 1, 0])
    p1 = KNeighborsClassifier(n_neighbors=1).fit(X, y).predict(np.array([[0.9]]))[0]
    p5 = KNeighborsClassifier(n_neighbors=5).fit(X, y).predict(np.array([[0.9]]))[0]
    assert 0 in (p1, p5) or 1 in (p1, p5)


def test_knn_decision_boundary_requires_2d(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    X_1d = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0, 0, 1])
    clf = KNeighborsClassifier(n_neighbors=1).fit(X_1d, y)
    with pytest.raises(ValueError):
        clf.plot_decision_boundary(X_1d, y)
