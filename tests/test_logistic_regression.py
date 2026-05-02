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


def test_logistic_regression_fit_returns_self():
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1])
    model = LogisticRegression(n_iterations=10)
    assert model.fit(X, y) is model


def test_logistic_regression_weights_and_bias_shape():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 4))
    y = (X[:, 0] > 0).astype(int)
    model = LogisticRegression(n_iterations=50).fit(X, y)
    assert model.weights.shape == (4,)
    assert isinstance(model.bias, float)


def test_logistic_regression_score_matches_manual_accuracy():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(40, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = LogisticRegression(learning_rate=0.1, n_iterations=300).fit(X, y)
    preds = model.predict(X)
    assert np.isclose(model.score(X, y), np.mean(preds == y))


def test_logistic_regression_probability_monotonicity():
    """For a 1-feature model, P(y=1|x) should be monotonically non-decreasing in x
    when the learned weight is positive (and vice versa)."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(60, 1))
    y = (X[:, 0] > 0).astype(int)
    model = LogisticRegression(learning_rate=0.1, n_iterations=500).fit(X, y)
    grid = np.linspace(-3, 3, 50).reshape(-1, 1)
    probs = model.predict_probability(grid)
    sign = 1 if model.weights[0] >= 0 else -1
    diffs = sign * np.diff(probs)
    assert np.all(diffs >= -1e-10)


def test_logistic_regression_probability_near_threshold():
    """Predictions should threshold at 0.5 exactly."""
    model = LogisticRegression(learning_rate=0.5, n_iterations=1000)
    X = np.array([[-5.0], [0.0], [5.0]])
    y = np.array([0, 0, 1])
    model.fit(X, y)
    probs = model.predict_probability(X)
    preds = model.predict(X)
    assert np.all((probs >= 0.5) == (preds == 1))


def test_logistic_regression_decreases_loss():
    """Check that loss after many iterations is lower than after very few."""
    rng = np.random.default_rng(4)
    X = np.vstack([rng.normal(loc=-1.5, size=(20, 2)),
                   rng.normal(loc=1.5, size=(20, 2))])
    y = np.array([0] * 20 + [1] * 20)

    def _loss(m):
        p = m.predict_probability(X)
        eps = 1e-12
        return -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

    short = LogisticRegression(learning_rate=0.1, n_iterations=2).fit(X, y)
    long = LogisticRegression(learning_rate=0.1, n_iterations=500).fit(X, y)
    assert _loss(long) < _loss(short)


def test_logistic_regression_sigmoid_extreme_stability():
    """Sigmoid is clipped so very large linear terms do not overflow."""
    X = np.array([[1000.0], [-1000.0]])
    y = np.array([1, 0])
    model = LogisticRegression(learning_rate=0.1, n_iterations=10).fit(X, y)
    probs = model.predict_probability(X)
    assert np.all(np.isfinite(probs))
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_logistic_regression_constant_feature_still_fits():
    """A feature that is constant gets a small-but-finite weight; model should still run."""
    rng = np.random.default_rng(8)
    X = np.column_stack([np.ones(30), rng.normal(size=30)])
    y = (X[:, 1] > 0).astype(int)
    model = LogisticRegression(learning_rate=0.1, n_iterations=200).fit(X, y)
    assert model.score(X, y) >= 0.8


def test_logistic_regression_default_hyperparameters():
    model = LogisticRegression()
    assert model.learning_rate == 0.01
    assert model.n_iterations == 1000
    assert model.weights is None
    assert model.bias is None


def test_logistic_regression_predict_returns_integers():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(10, 2))
    y = (X[:, 0] > 0).astype(int)
    model = LogisticRegression(n_iterations=20).fit(X, y)
    preds = model.predict(X)
    assert np.issubdtype(preds.dtype, np.integer)
