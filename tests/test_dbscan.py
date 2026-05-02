import numpy as np
import pytest

from mlpackage.unsupervised_learning import DBSCANClustering


def test_dbscan_finds_two_clusters():
    rng = np.random.default_rng(0)
    cluster_a = rng.normal(loc=[0, 0], scale=0.3, size=(30, 2))
    cluster_b = rng.normal(loc=[5, 5], scale=0.3, size=(30, 2))
    X = np.vstack([cluster_a, cluster_b])
    model = DBSCANClustering(epsilon=1.0, min_neighbours=3)
    labels = model.fit_predict(X)
    assert labels.shape == (60,)
    unique = set(labels)
    unique.discard(-1)
    assert len(unique) == 2


def test_dbscan_noise_points():
    rng = np.random.default_rng(1)
    cluster = rng.normal(loc=[0, 0], scale=0.2, size=(20, 2))
    outlier = np.array([[100.0, 100.0]])
    X = np.vstack([cluster, outlier])
    model = DBSCANClustering(epsilon=0.8, min_neighbours=3)
    model.fit(X)
    assert model.labels_[-1] == -1


def test_dbscan_all_noise_when_eps_tiny():
    X = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
    model = DBSCANClustering(epsilon=0.001, min_neighbours=2)
    labels = model.fit_predict(X)
    assert np.all(labels == -1)


def test_dbscan_single_cluster():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(25, 2)) * 0.1
    model = DBSCANClustering(epsilon=1.0, min_neighbours=2)
    labels = model.fit_predict(X)
    unique = set(labels)
    unique.discard(-1)
    assert len(unique) == 1


def test_dbscan_empty_data():
    model = DBSCANClustering()
    with pytest.raises(ValueError):
        model.fit(np.empty((0, 2)))


def test_dbscan_labels_attribute_set_after_fit():
    X = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]])
    model = DBSCANClustering(epsilon=0.5, min_neighbours=2)
    assert model.labels_ is None
    model.fit(X)
    assert model.labels_ is not None
    assert model.labels_.shape == (3,)


def test_dbscan_fit_returns_self():
    X = np.array([[0.0, 0.0], [0.1, 0.0]])
    model = DBSCANClustering(epsilon=0.5, min_neighbours=1)
    assert model.fit(X) is model


def test_dbscan_fit_predict_equals_labels_attr():
    X = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])
    model = DBSCANClustering(epsilon=0.5, min_neighbours=2)
    labels_returned = model.fit_predict(X)
    assert np.array_equal(labels_returned, model.labels_)


def test_dbscan_defaults():
    model = DBSCANClustering()
    assert model.epsilon == 0.5
    assert model.min_neighbours == 5
    assert model.labels_ is None


def test_dbscan_labels_are_contiguous_integers():
    """Cluster IDs should start at 0 and increment without gaps."""
    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.normal(loc=[0, 0], scale=0.2, size=(15, 2)),
        rng.normal(loc=[10, 0], scale=0.2, size=(15, 2)),
        rng.normal(loc=[0, 10], scale=0.2, size=(15, 2)),
    ])
    model = DBSCANClustering(epsilon=1.0, min_neighbours=3).fit(X)
    positive_labels = np.sort(np.unique(model.labels_[model.labels_ != -1]))
    assert positive_labels.tolist() == list(range(len(positive_labels)))


def test_dbscan_three_clusters_detected():
    rng = np.random.default_rng(1)
    c0 = rng.normal(loc=[0, 0], scale=0.2, size=(20, 2))
    c1 = rng.normal(loc=[10, 0], scale=0.2, size=(20, 2))
    c2 = rng.normal(loc=[0, 10], scale=0.2, size=(20, 2))
    X = np.vstack([c0, c1, c2])
    model = DBSCANClustering(epsilon=1.0, min_neighbours=3).fit(X)
    unique = set(model.labels_)
    unique.discard(-1)
    assert len(unique) == 3


def test_dbscan_labels_size_matches_input():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(35, 3))
    model = DBSCANClustering(epsilon=1.0, min_neighbours=3).fit(X)
    assert model.labels_.shape == (35,)


def test_dbscan_only_labels_are_minus1_or_nonnegative():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 2))
    model = DBSCANClustering(epsilon=0.3, min_neighbours=3).fit(X)
    assert np.all((model.labels_ == -1) | (model.labels_ >= 0))


def test_dbscan_points_within_same_tight_cluster_share_label():
    """All points within one very tight cluster should receive the same non-noise label."""
    rng = np.random.default_rng(1)
    X = rng.normal(loc=[0, 0], scale=0.05, size=(20, 2))
    model = DBSCANClustering(epsilon=0.5, min_neighbours=3).fit(X)
    labels = model.labels_[model.labels_ != -1]
    assert labels.size == 20
    assert len(np.unique(labels)) == 1


def test_dbscan_large_epsilon_one_cluster():
    """Huge epsilon should merge everything into a single cluster."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(25, 2))
    model = DBSCANClustering(epsilon=1000.0, min_neighbours=2).fit(X)
    unique = set(model.labels_)
    unique.discard(-1)
    assert len(unique) == 1


def test_dbscan_min_neighbours_affects_clustering():
    """Increasing min_neighbours should increase (or keep equal) the number of noise points."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(30, 2))
    noise_low = np.sum(
        DBSCANClustering(epsilon=0.5, min_neighbours=2).fit(X).labels_ == -1
    )
    noise_high = np.sum(
        DBSCANClustering(epsilon=0.5, min_neighbours=10).fit(X).labels_ == -1
    )
    assert noise_high >= noise_low
