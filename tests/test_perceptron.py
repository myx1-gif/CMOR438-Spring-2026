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


def test_perceptron_fit_returns_self():
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1])
    model = Perceptron(max_iter=3)
    assert model.fit(X, y) is model


def test_perceptron_coef_and_intercept_shapes():
    X = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    y = np.array([0, 0, 1, 0])
    model = Perceptron(lr=0.1, max_iter=50).fit(X, y)
    assert model.coef_.shape == (2,)
    assert isinstance(model.intercept_, float)


def test_perceptron_learns_logical_and():
    """Perceptron should learn the AND function exactly."""
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 0, 0, 1])
    model = Perceptron(lr=0.1, max_iter=200).fit(X, y)
    assert np.array_equal(model.predict(X), y)


def test_perceptron_learns_logical_or():
    """Perceptron should learn the OR function exactly."""
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1, 1, 1])
    model = Perceptron(lr=0.1, max_iter=200).fit(X, y)
    assert np.array_equal(model.predict(X), y)


def test_perceptron_final_training_error_nonincreasing_trend():
    """On separable data, error curve end should be no greater than the start."""
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(loc=-2, size=(40, 2)),
                   rng.normal(loc=2, size=(40, 2))])
    y = np.array([0] * 40 + [1] * 40)
    model = Perceptron(lr=0.1, max_iter=100).fit(X, y)
    assert model.training_errors[-1] <= model.training_errors[0]


def test_perceptron_score_matches_manual():
    rng = np.random.default_rng(2)
    X = np.vstack([rng.normal(loc=-1.5, size=(15, 2)),
                   rng.normal(loc=1.5, size=(15, 2))])
    y = np.array([0] * 15 + [1] * 15)
    model = Perceptron(lr=0.1, max_iter=100).fit(X, y)
    preds = model.predict(X)
    assert np.isclose(model.score(X, y), float(np.mean(preds == y)))


def test_perceptron_predict_output_values_only_0_or_1():
    """Step-function activation must produce only 0 or 1."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(25, 3))
    y = (X[:, 0] > 0).astype(int)
    model = Perceptron(lr=0.1, max_iter=50).fit(X, y)
    out = model.predict(rng.normal(size=(50, 3)))
    assert set(np.unique(out)).issubset({0, 1})


def test_perceptron_default_hyperparameters():
    model = Perceptron()
    assert model.lr == 0.01
    assert model.max_iter == 1000
    assert model.coef_ is None
    assert model.intercept_ is None
    assert model.training_errors == []


def test_perceptron_refit_resets_training_errors():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    model = Perceptron(lr=0.1, max_iter=10).fit(X, y)
    assert len(model.training_errors) == 10
    model.fit(X, y)
    assert len(model.training_errors) == 10


def test_perceptron_plot_empty_errors_no_raise(monkeypatch):
    """plot_training_loss should not raise when training_errors is empty."""
    monkeypatch.setattr(plt, "show", lambda: None)
    model = Perceptron()
    model.plot_training_loss()
    plt.close("all")


def test_perceptron_score_before_fit_raises():
    model = Perceptron()
    with pytest.raises(AttributeError):
        model.score(np.zeros((1, 2)), np.zeros(1))
