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
