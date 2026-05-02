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


def test_linear_regression_univariate_slope_intercept_formula():
    """For a simple 1D line, check coefficients match the closed-form formulas."""
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([2.0, 3.9, 6.1, 7.9, 10.2])
    model = LinearRegression().fit(X, y)

    x_flat = X.ravel()
    slope_expected = np.sum((x_flat - x_flat.mean()) * (y - y.mean())) / np.sum(
        (x_flat - x_flat.mean()) ** 2
    )
    intercept_expected = y.mean() - slope_expected * x_flat.mean()

    assert np.isclose(model.coef_[0], slope_expected, rtol=1e-8)
    assert np.isclose(model.intercept, intercept_expected, rtol=1e-8)


def test_linear_regression_not_fitted_initially():
    model = LinearRegression()
    assert not model.fitted
    assert model.coef_ is None
    assert model.intercept is None
    assert model.coefficients is None


def test_linear_regression_fit_returns_self():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([1.0, 2.0, 3.0])
    model = LinearRegression()
    result = model.fit(X, y)
    assert result is model


def test_linear_regression_predict_shape():
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(20, 4))
    y_train = rng.normal(size=20)
    X_test = rng.normal(size=(7, 4))
    model = LinearRegression().fit(X_train, y_train)
    preds = model.predict(X_test)
    assert preds.shape == (7,)


def test_linear_regression_rmse_computation():
    """RMSE should equal the formula sqrt(mean((y - y_hat)^2))."""
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 2.0, 4.0])
    model = LinearRegression().fit(X, y)
    preds = model.predict(X)
    expected_rmse = float(np.sqrt(np.mean((y - preds) ** 2)))
    assert np.isclose(model.rmse(X, y), expected_rmse, rtol=1e-10)


def test_linear_regression_rmse_before_fit_raises():
    model = LinearRegression()
    with pytest.raises(AttributeError):
        model.rmse(np.array([[1.0]]), np.array([1.0]))


def test_linear_regression_r2_before_fit_raises():
    model = LinearRegression()
    with pytest.raises(AttributeError):
        model.R_squared(np.array([[1.0]]), np.array([1.0]))


def test_linear_regression_r2_in_valid_range():
    """R² on training data for well-fit data should be between 0 and 1."""
    rng = np.random.default_rng(11)
    X = rng.normal(size=(50, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + 3.0 + 0.1 * rng.normal(size=50)
    model = LinearRegression().fit(X, y)
    r2 = model.R_squared(X, y)
    assert 0.9 <= r2 <= 1.0


def test_linear_regression_handles_multiple_features():
    """Verify recovery of true parameters with many features."""
    rng = np.random.default_rng(2026)
    n, d = 200, 8
    X = rng.normal(size=(n, d))
    true_coef = rng.normal(size=d)
    true_intercept = 1.7
    y = true_intercept + X @ true_coef
    model = LinearRegression().fit(X, y)
    assert np.allclose(model.coef_, true_coef, atol=1e-8)
    assert np.isclose(model.intercept, true_intercept, atol=1e-8)


def test_linear_regression_integer_inputs_cast_to_float():
    """Integer feature arrays should be converted internally (no dtype error)."""
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=int)
    y = np.array([3, 7, 11], dtype=int)
    model = LinearRegression().fit(X, y)
    preds = model.predict(X)
    assert np.allclose(preds, y, atol=1e-8)


def test_linear_regression_list_input_supported():
    """fit/predict should accept Python lists (converted via np.asarray)."""
    X = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]
    y = [3.0, 5.0, 7.0, 9.0]
    model = LinearRegression().fit(X, y)
    pred = model.predict([[5.0, 6.0]])
    assert pred.shape == (1,)
    assert np.isclose(pred[0], 11.0, atol=1e-8)


def test_linear_regression_refit_clears_previous_state():
    """Refitting on new data should replace previous coefficients."""
    X1 = np.array([[0.0], [1.0], [2.0]])
    y1 = np.array([0.0, 1.0, 2.0])
    X2 = np.array([[0.0], [1.0], [2.0]])
    y2 = np.array([0.0, 2.0, 4.0])
    model = LinearRegression().fit(X1, y1)
    first_coef = model.coef_.copy()
    model.fit(X2, y2)
    assert not np.allclose(first_coef, model.coef_)
    assert np.isclose(model.coef_[0], 2.0, atol=1e-8)


def test_linear_regression_2d_target_ravels():
    """y provided as a column vector should still fit."""
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([[0.0], [2.0], [4.0], [6.0]])
    model = LinearRegression().fit(X, y)
    assert np.isclose(model.coef_[0], 2.0, atol=1e-8)
    assert np.isclose(model.intercept, 0.0, atol=1e-8)
