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


def test_decision_tree_fit_returns_self():
    X = np.array([[0.0], [1.0]])
    y = np.array([0, 1])
    clf = DecisionTreeClassifier()
    assert clf.fit(X, y) is clf


def test_decision_tree_unlimited_depth_fits_training_data_exactly():
    """Without a depth cap, tree should reach 100% train accuracy on distinct samples."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    clf = DecisionTreeClassifier(max_depth=None).fit(X, y)
    assert clf.score(X, y) == 1.0


def test_decision_tree_multi_class_classification():
    X = np.array([
        [0.0], [0.1], [0.2],
        [5.0], [5.1], [5.2],
        [10.0], [10.1], [10.2],
    ])
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    clf = DecisionTreeClassifier().fit(X, y)
    assert np.array_equal(clf.predict(X), y)
    assert set(np.unique(clf.predict(X))) == {0, 1, 2}


def test_decision_tree_feature_selection_prefers_informative_feature():
    """When one feature is noise and another is perfectly separating,
    predictions should match the separating feature's rule."""
    rng = np.random.default_rng(0)
    X_info = np.arange(20).reshape(-1, 1).astype(float)
    X_noise = rng.normal(size=(20, 1))
    X = np.hstack([X_noise, X_info])
    y = (X_info.ravel() > 9).astype(int)
    clf = DecisionTreeClassifier(max_depth=2).fit(X, y)
    assert clf.score(X, y) == 1.0


def test_decision_tree_predict_on_unseen_data_shape():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(20, 3))
    y = (X[:, 0] > 0).astype(int)
    clf = DecisionTreeClassifier(max_depth=3).fit(X, y)
    preds = clf.predict(rng.normal(size=(8, 3)))
    assert preds.shape == (8,)


def test_decision_tree_deterministic_predictions():
    """Fitting twice on the same data must produce the same predictions."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 2))
    y = (X[:, 0] > 0).astype(int)
    clf1 = DecisionTreeClassifier(max_depth=3).fit(X, y)
    clf2 = DecisionTreeClassifier(max_depth=3).fit(X, y)
    assert np.array_equal(clf1.predict(X), clf2.predict(X))


def test_decision_tree_duplicate_feature_values():
    """Tree should not crash when feature values are all the same."""
    X = np.zeros((5, 2))
    y = np.array([0, 1, 0, 1, 0])
    clf = DecisionTreeClassifier().fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (5,)
    assert set(preds) <= {0, 1}


def test_random_forest_requires_fit():
    forest = RandomForestClassifier()
    with pytest.raises(AttributeError):
        forest.predict(np.zeros((1, 2)))


def test_random_forest_multi_class_support():
    rng = np.random.default_rng(3)
    c0 = rng.normal(loc=[0, 0], scale=0.3, size=(15, 2))
    c1 = rng.normal(loc=[5, 0], scale=0.3, size=(15, 2))
    c2 = rng.normal(loc=[0, 5], scale=0.3, size=(15, 2))
    X = np.vstack([c0, c1, c2])
    y = np.array([0] * 15 + [1] * 15 + [2] * 15)
    np.random.seed(1)
    forest = RandomForestClassifier(n_estimators=10, max_depth=4, max_features=None)
    forest.fit(X, y)
    preds = forest.predict(X)
    assert set(np.unique(preds)).issubset({0, 1, 2})
    assert np.mean(preds == y) >= 0.9


def test_random_forest_default_trees_count():
    forest = RandomForestClassifier()
    assert forest.n_estimators == 100
    assert forest.max_features == "sqrt"


def test_random_forest_invalid_max_features():
    X = np.array([[0.0], [1.0]])
    y = np.array([0, 1])
    forest = RandomForestClassifier(n_estimators=1, max_features="invalid")
    with pytest.raises(ValueError):
        forest.fit(X, y)


def test_random_forest_predict_shape_matches_input():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(20, 3))
    y = (X[:, 0] > 0).astype(int)
    np.random.seed(11)
    forest = RandomForestClassifier(n_estimators=7, max_depth=3).fit(X, y)
    preds = forest.predict(rng.normal(size=(6, 3)))
    assert preds.shape == (6,)
