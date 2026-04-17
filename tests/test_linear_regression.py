import numpy as np
import pytest

from mlpackage.supervised_learning import LinearRegression


def test_linear_regression_perfect_line():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    true_intercept = 2.5
    true_coef = np.array([1.0, -0.5, 0.25])
    y = true_intercept + X @ true_coef

    model = LinearRegression()
    model.fit(X, y)
    assert model.fitted
    assert np.allclose(model.intercept, true_intercept, rtol=1e-10)
    assert np.allclose(model.coefficients, true_coef, rtol=1e-10)
    assert np.allclose(model.coef_, true_coef, rtol=1e-10)
    preds = model.predict(X)
    assert np.allclose(preds, y)
    assert model.rmse(X, y) < 1e-9
    assert model.R_squared(X, y) > 1.0 - 1e-12


def test_linear_regression_predict_before_fit():
    model = LinearRegression()
    with pytest.raises(AttributeError):
        model.predict(np.zeros((2, 2)))


def test_linear_regression_empty_fit():
    model = LinearRegression()
    with pytest.raises(ValueError):
        model.fit(np.empty((0, 2)), np.array([]))


def test_linear_regression_shape_mismatch():
    model = LinearRegression()
    with pytest.raises(ValueError):
        model.fit(np.zeros((3, 2)), np.zeros(2))


def test_linear_regression_r2_constant_targets():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([5.0, 5.0, 5.0])
    model = LinearRegression()
    model.fit(X, y)
    assert model.R_squared(X, y) == 1.0
