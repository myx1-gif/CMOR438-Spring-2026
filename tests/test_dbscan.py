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
