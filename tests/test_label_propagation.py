import numpy as np
import pytest

from mlpackage.unsupervised_learning import GraphLabelPropagation


def _make_two_class_data():
    """Two tight clusters with a few labelled anchors and unlabelled points."""
    rng = np.random.default_rng(0)
    cluster_0 = rng.normal(loc=[0, 0], scale=0.3, size=(15, 2))
    cluster_1 = rng.normal(loc=[5, 5], scale=0.3, size=(15, 2))
    X = np.vstack([cluster_0, cluster_1])
    y = np.full(30, -1, dtype=int)
    y[0] = 0
    y[1] = 0
    y[15] = 1
    y[16] = 1
    return X, y


def test_propagation_labels_all_points():
    X, y = _make_two_class_data()
    model = GraphLabelPropagation(spread=0.9, bandwidth=1.0).fit(X, y)
    preds = model.predict()
    assert preds.shape == (30,)
    assert set(preds).issubset({0, 1})


def test_propagation_correct_clusters():
    X, y = _make_two_class_data()
    model = GraphLabelPropagation(spread=0.9, bandwidth=1.0).fit(X, y)
    preds = model.predict()
    assert np.all(preds[:15] == 0)
    assert np.all(preds[15:] == 1)


def test_clamped_labels_preserved():
    X, y = _make_two_class_data()
    model = GraphLabelPropagation(clamp_labelled=True).fit(X, y)
    preds = model.predict()
    assert preds[0] == 0
    assert preds[1] == 0
    assert preds[15] == 1
    assert preds[16] == 1


def test_distribution_rows_are_nonnegative():
    X, y = _make_two_class_data()
    model = GraphLabelPropagation().fit(X, y)
    assert np.all(model.distribution_ >= 0)


def test_multiclass_propagation():
    rng = np.random.default_rng(1)
    c0 = rng.normal(loc=[0, 0], scale=0.2, size=(10, 2))
    c1 = rng.normal(loc=[5, 0], scale=0.2, size=(10, 2))
    c2 = rng.normal(loc=[2.5, 5], scale=0.2, size=(10, 2))
    X = np.vstack([c0, c1, c2])
    y = np.full(30, -1, dtype=int)
    y[0] = 0
    y[10] = 1
    y[20] = 2
    model = GraphLabelPropagation(bandwidth=1.0).fit(X, y)
    preds = model.predict()
    assert set(preds) == {0, 1, 2}


def test_predict_before_fit_raises():
    model = GraphLabelPropagation()
    with pytest.raises(AttributeError):
        model.predict()


def test_empty_data_raises():
    model = GraphLabelPropagation()
    with pytest.raises(ValueError):
        model.fit(np.empty((0, 2)), np.array([], dtype=int))


def test_mismatched_lengths_raises():
    model = GraphLabelPropagation()
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0])
    with pytest.raises(ValueError):
        model.fit(X, y)
