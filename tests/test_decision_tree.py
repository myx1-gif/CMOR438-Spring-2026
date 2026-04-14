import numpy as np
import pytest

from mlpackage.supervised_learning import DecisionTreeClassifier, RandomForestClassifier


def test_decision_tree_perfect_split_on_intermediate_points():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    clf = DecisionTreeClassifier(max_depth=2)
    clf.fit(X, y)

    preds = clf.predict(np.array([[0.5], [2.5]]))
    assert np.array_equal(preds, np.array([0, 1]))


def test_decision_tree_single_class_predictions():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([1, 1, 1])
    clf = DecisionTreeClassifier()
    clf.fit(X, y)

    preds = clf.predict(np.array([[0.0], [10.0]]))
    assert np.all(preds == 1)


def test_decision_tree_max_depth_zero_predicts_majority_class():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    clf = DecisionTreeClassifier(max_depth=0)
    clf.fit(X, y)

    preds = clf.predict(X)
    assert np.all(preds == 0)


def test_decision_tree_predict_unseen_values_shape():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X, y)

    preds = clf.predict(np.array([[10.0], [-5.0]]))
    assert preds.shape == (2,)


def test_decision_tree_xor_like_grid():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 0, 1, 1])
    clf = DecisionTreeClassifier(max_depth=2)
    clf.fit(X, y)
    assert np.array_equal(clf.predict(X), y)
    assert clf.score(X, y) == 1.0


def test_decision_tree_requires_fit():
    clf = DecisionTreeClassifier()
    with pytest.raises(AttributeError):
        clf.predict(np.zeros((1, 2)))


def test_decision_tree_empty_inputs():
    clf = DecisionTreeClassifier()
    with pytest.raises(ValueError):
        clf.fit(np.empty((0, 2)), np.array([]))
    with pytest.raises(ValueError):
        clf.fit(np.zeros((3, 2)), np.zeros(2))


def test_random_forest_shape_and_determinism():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    np.random.seed(42)
    forest = RandomForestClassifier(n_estimators=15, max_depth=3, max_features="sqrt")
    forest.fit(X, y)
    pred = forest.predict(X)
    assert pred.shape == (40,)

    np.random.seed(42)
    forest2 = RandomForestClassifier(n_estimators=15, max_depth=3, max_features="sqrt")
    forest2.fit(X, y)
    assert np.array_equal(forest.predict(X), forest2.predict(X))


def test_random_forest_all_features_mode():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    np.random.seed(0)
    forest = RandomForestClassifier(n_estimators=30, max_depth=4, max_features=None)
    forest.fit(X, y)
    assert forest.predict(X).shape == (4,)
