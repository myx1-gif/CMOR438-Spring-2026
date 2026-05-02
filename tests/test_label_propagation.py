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


def test_label_propagation_fit_returns_self():
    X, y = _make_two_class_data()
    model = GraphLabelPropagation()
    assert model.fit(X, y) is model


def test_label_propagation_defaults():
    model = GraphLabelPropagation()
    assert model.spread == 0.9
    assert model.bandwidth == 1.0
    assert model.max_iterations == 1000
    assert model.convergence_tol == 1e-4
    assert model.clamp_labelled is True
    assert model.classes_ is None
    assert model.labels_ is None
    assert model.distribution_ is None
    assert model.transition_ is None


def test_label_propagation_transition_rows_sum_to_one():
    """Each row of the transition matrix should sum to 1 (row-normalisation)."""
    X, y = _make_two_class_data()
    model = GraphLabelPropagation().fit(X, y)
    row_sums = model.transition_.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-10)


def test_label_propagation_transition_zero_diagonal():
    """RBF graph removes self-similarity, so the transition matrix has a zero diagonal."""
    X, y = _make_two_class_data()
    model = GraphLabelPropagation().fit(X, y)
    assert np.allclose(np.diag(model.transition_), 0.0, atol=1e-12)


def test_label_propagation_distribution_shape():
    X, y = _make_two_class_data()
    model = GraphLabelPropagation().fit(X, y)
    assert model.distribution_.shape == (30, 2)
    assert model.labels_.shape == (30,)


def test_label_propagation_classes_stored_sorted():
    """classes_ should be a sorted unique list of observed labels (ignoring -1)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 2))
    y = np.array([-1] * 17 + [5, 2, 2])
    model = GraphLabelPropagation().fit(X, y)
    assert list(model.classes_) == [2, 5]


def test_label_propagation_deterministic():
    """Same inputs should yield identical predictions (no randomness)."""
    X, y = _make_two_class_data()
    model_a = GraphLabelPropagation(spread=0.8, bandwidth=1.5).fit(X, y)
    model_b = GraphLabelPropagation(spread=0.8, bandwidth=1.5).fit(X, y)
    assert np.array_equal(model_a.labels_, model_b.labels_)
    assert np.allclose(model_a.distribution_, model_b.distribution_, atol=1e-12)


def test_label_propagation_clamped_vs_unclamped():
    """With clamp_labelled=True, labelled rows' distribution stays one-hot."""
    X, y = _make_two_class_data()
    model = GraphLabelPropagation(clamp_labelled=True).fit(X, y)
    clamped_idx = np.where(y != -1)[0]
    for i in clamped_idx:
        col = int(np.argmax(model.distribution_[i]))
        assert model.distribution_[i, col] == 1.0
        assert np.sum(model.distribution_[i]) == 1.0


def test_label_propagation_single_labelled_class():
    """If only one class is labelled, all predictions should be that class."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(15, 2))
    y = np.full(15, -1, dtype=int)
    y[0] = 4
    model = GraphLabelPropagation(bandwidth=1.0).fit(X, y)
    assert np.all(model.labels_ == 4)


def test_label_propagation_labelled_points_predicted_correctly():
    """All originally-labelled points should keep their true class."""
    X, y = _make_two_class_data()
    model = GraphLabelPropagation().fit(X, y)
    labelled_mask = y != -1
    assert np.array_equal(model.labels_[labelled_mask], y[labelled_mask])


def test_label_propagation_predict_returns_same_shape():
    X, y = _make_two_class_data()
    model = GraphLabelPropagation().fit(X, y)
    preds = model.predict()
    assert preds.shape == (30,)


def test_label_propagation_bandwidth_affects_similarity():
    """Larger bandwidth should produce a smoother (less peaked) similarity matrix."""
    X, _ = _make_two_class_data()
    y = np.full(X.shape[0], -1, dtype=int)
    y[0] = 0
    y[15] = 1
    m_small = GraphLabelPropagation(bandwidth=0.1).fit(X, y)
    m_large = GraphLabelPropagation(bandwidth=5.0).fit(X, y)
    assert m_small.transition_.var() >= m_large.transition_.var()
