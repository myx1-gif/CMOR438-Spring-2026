import numpy as np
import pytest

from mlpackage.supervised_learning import LogisticRegression


def test_logistic_regression_linearly_separable():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(loc=-2, size=(30, 2)),
                    rng.normal(loc=2, size=(30, 2))])
    y = np.array([0] * 30 + [1] * 30)
    model = LogisticRegression(learning_rate=0.1, n_iterations=500)
    model.fit(X, y)
    assert model.score(X, y) >= 0.95


def test_logistic_regression_predict_proba_range():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 3))
    y = (X[:, 0] > 0).astype(int)
    model = LogisticRegression(learning_rate=0.05, n_iterations=300)
    model.fit(X, y)
    proba = model.predict_probability(X)
    assert proba.shape == (40,)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)


def test_logistic_regression_predict_binary_output():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(20, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = LogisticRegression(learning_rate=0.1, n_iterations=200)
    model.fit(X, y)
    preds = model.predict(X)
    assert set(np.unique(preds)).issubset({0, 1})


def test_logistic_regression_predict_before_fit():
    model = LogisticRegression()
    with pytest.raises(AttributeError):
        model.predict(np.zeros((2, 2)))


def test_logistic_regression_predict_probability_before_fit():
    model = LogisticRegression()
    with pytest.raises(AttributeError):
        model.predict_probability(np.zeros((2, 2)))


def test_logistic_regression_empty_fit():
    model = LogisticRegression()
    with pytest.raises(ValueError):
        model.fit(np.empty((0, 2)), np.array([]))


def test_logistic_regression_shape_mismatch():
    model = LogisticRegression()
    with pytest.raises(ValueError):
        model.fit(np.zeros((3, 2)), np.zeros(2))
