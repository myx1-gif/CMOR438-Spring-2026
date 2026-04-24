import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from mlpackage.supervised_learning import Perceptron


def test_perceptron_linearly_separable():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(loc=-2, size=(30, 2)),
                    rng.normal(loc=2, size=(30, 2))])
    y = np.array([0] * 30 + [1] * 30)
    model = Perceptron(lr=0.1, max_iter=200)
    model.fit(X, y)
    assert model.score(X, y) >= 0.95


def test_perceptron_predict_binary_only():
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    y = np.array([0, 0, 0, 1])
    model = Perceptron(lr=0.1, max_iter=100)
    model.fit(X, y)
    preds = model.predict(X)
    assert set(np.unique(preds)).issubset({0, 1})


def test_perceptron_training_errors_recorded():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    model = Perceptron(lr=0.1, max_iter=50)
    model.fit(X, y)
    assert len(model.training_errors) == 50


def test_perceptron_confusion_matrix_shape():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(loc=-2, size=(20, 2)),
                    rng.normal(loc=2, size=(20, 2))])
    y = np.array([0] * 20 + [1] * 20)
    model = Perceptron(lr=0.1, max_iter=100)
    model.fit(X, y)
    cm = model.confusion_matrix(X, y)
    assert cm.shape == (2, 2)


def test_perceptron_predict_before_fit():
    model = Perceptron()
    with pytest.raises(AttributeError):
        model.predict(np.zeros((2, 2)))


def test_perceptron_empty_fit():
    model = Perceptron()
    with pytest.raises(ValueError):
        model.fit(np.empty((0, 2)), np.array([]))


def test_perceptron_shape_mismatch():
    model = Perceptron()
    with pytest.raises(ValueError):
        model.fit(np.zeros((5, 2)), np.zeros(3))


def test_perceptron_plot_loss_runs(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    model = Perceptron(lr=0.1, max_iter=10)
    model.fit(X, y)
    model.plot_training_loss()
    plt.close("all")
