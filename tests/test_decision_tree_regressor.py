import numpy as np
import pytest

from mlpackage.supervised_learning import DecisionTreeRegressor, RandomForestRegressor


def test_decision_tree_regressor_perfect_split_between_points():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    model = DecisionTreeRegressor(max_depth=2)
    model.fit(X, y)

    preds = model.predict(np.array([[0.5], [2.5]]))
    assert np.allclose(preds, np.array([0.0, 1.0]))


def test_decision_tree_regressor_single_repeated_target_value():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([5.0, 5.0, 5.0])
    model = DecisionTreeRegressor()
    model.fit(X, y)

    preds = model.predict(np.array([[0.0], [10.0]]))
    assert np.allclose(preds, 5.0)


def test_decision_tree_regressor_max_depth_zero_returns_mean():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    model = DecisionTreeRegressor(max_depth=0)
    model.fit(X, y)

    preds = model.predict(X)
    assert np.allclose(preds, np.mean(y))


def test_decision_tree_regressor_predict_unseen_values_range_and_shape():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    model = DecisionTreeRegressor(max_depth=3)
    model.fit(X, y)

    preds = model.predict(np.array([[10.0], [-5.0]]))
    assert preds.shape == (2,)
    assert np.all(preds >= 0.0) and np.all(preds <= 1.0)


def test_decision_tree_regressor_reproduces_training_points():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.1, 0.9, 2.1, 3.0])
    model = DecisionTreeRegressor(max_depth=2)
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == (4,)
    assert np.allclose(pred, y)
    assert model.score(X, y) > 0.99


def test_decision_tree_regressor_requires_fit():
    model = DecisionTreeRegressor()
    with pytest.raises(AttributeError):
        model.predict(np.array([[0.0]]))


def test_decision_tree_regressor_input_validation():
    model = DecisionTreeRegressor()
    with pytest.raises(ValueError):
        model.fit(np.empty((0, 2)), np.array([]))
    with pytest.raises(ValueError):
        model.fit(np.zeros((3, 2)), np.zeros(2))


def test_random_forest_regressor_predict_shape_and_seed_repeatability():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(50, 4))
    y = 1.5 * X[:, 0] - 0.5 * X[:, 1] + 0.1 * rng.normal(size=50)

    np.random.seed(7)
    forest1 = RandomForestRegressor(n_estimators=20, max_depth=4, max_features="sqrt")
    forest1.fit(X, y)
    pred1 = forest1.predict(X)
    assert pred1.shape == (50,)

    np.random.seed(7)
    forest2 = RandomForestRegressor(n_estimators=20, max_depth=4, max_features="sqrt")
    forest2.fit(X, y)
    pred2 = forest2.predict(X)
    assert np.allclose(pred1, pred2)


def test_random_forest_regressor_all_features_mode():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.2, 0.8, 2.2, 2.9])
    np.random.seed(0)
    model = RandomForestRegressor(n_estimators=25, max_depth=3, max_features=None)
    model.fit(X, y)
    assert model.predict(X).shape == (4,)


def test_decision_tree_regressor_fit_returns_self():
    X = np.array([[0.0], [1.0]])
    y = np.array([0.0, 1.0])
    model = DecisionTreeRegressor()
    assert model.fit(X, y) is model


def test_decision_tree_regressor_max_depth_one_splits_once():
    """With max_depth=1 tree should produce exactly two distinct leaf values."""
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([0.0, 1.0, 2.0, 10.0, 11.0, 12.0])
    model = DecisionTreeRegressor(max_depth=1).fit(X, y)
    preds = model.predict(X)
    assert len(np.unique(preds)) == 2


def test_decision_tree_regressor_leaf_value_is_mean():
    """Prediction at a leaf is the mean of training y values that reached it."""
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([2.0, 4.0, 6.0, 8.0])
    model = DecisionTreeRegressor(max_depth=0).fit(X, y)
    assert np.isclose(model.predict(np.array([[0.0]]))[0], float(np.mean(y)))


def test_decision_tree_regressor_score_matches_R2_formula():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 2))
    y = X[:, 0] + 0.1 * rng.normal(size=30)
    model = DecisionTreeRegressor(max_depth=3).fit(X, y)
    preds = model.predict(X)
    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    expected = 1.0 - ss_res / ss_tot
    assert np.isclose(model.score(X, y), expected, rtol=1e-10)


def test_decision_tree_regressor_score_is_one_for_constant_target():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([5.0, 5.0, 5.0])
    model = DecisionTreeRegressor().fit(X, y)
    assert model.score(X, y) == 1.0


def test_decision_tree_regressor_piecewise_step_function():
    """Tree should recover a two-level step function exactly at sufficient depth."""
    X = np.linspace(0, 10, 40).reshape(-1, 1)
    y = np.where(X.ravel() < 5, -1.0, 3.0)
    model = DecisionTreeRegressor(max_depth=3).fit(X, y)
    assert np.allclose(model.predict(X), y, atol=1e-10)


def test_decision_tree_regressor_deterministic_predictions():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(20, 2))
    y = rng.normal(size=20)
    preds1 = DecisionTreeRegressor(max_depth=3).fit(X, y).predict(X)
    preds2 = DecisionTreeRegressor(max_depth=3).fit(X, y).predict(X)
    assert np.allclose(preds1, preds2)


def test_random_forest_regressor_requires_fit():
    forest = RandomForestRegressor()
    with pytest.raises(AttributeError):
        forest.predict(np.zeros((1, 2)))


def test_random_forest_regressor_input_validation():
    forest = RandomForestRegressor(n_estimators=2)
    with pytest.raises(ValueError):
        forest.fit(np.empty((0, 2)), np.array([]))
    with pytest.raises(ValueError):
        forest.fit(np.zeros((3, 2)), np.zeros(4))


def test_random_forest_regressor_invalid_max_features():
    forest = RandomForestRegressor(n_estimators=1, max_features="bogus")
    with pytest.raises(ValueError):
        forest.fit(np.array([[0.0], [1.0]]), np.array([0.0, 1.0]))


def test_random_forest_regressor_reasonable_fit():
    """On smooth data, ensemble prediction should correlate positively with the target."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 3))
    y = X[:, 0] * 2.0 - X[:, 1] * 0.5
    np.random.seed(3)
    forest = RandomForestRegressor(n_estimators=25, max_depth=5, max_features=None)
    forest.fit(X, y)
    preds = forest.predict(X)
    corr = np.corrcoef(preds, y)[0, 1]
    assert corr > 0.8


def test_random_forest_regressor_prediction_output_type():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0.0, 1.0, 2.0])
    np.random.seed(5)
    forest = RandomForestRegressor(n_estimators=3, max_features=None).fit(X, y)
    preds = forest.predict(X)
    assert np.issubdtype(preds.dtype, np.floating)


def test_random_forest_regressor_defaults():
    forest = RandomForestRegressor()
    assert forest.n_estimators == 100
    assert forest.max_features == "sqrt"
