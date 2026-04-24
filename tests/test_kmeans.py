import numpy as np
import pytest

from mlpackage.unsupervised_learning import KMeansClustering


def test_kmeans_finds_two_clusters():
    rng = np.random.default_rng(0)
    blob_a = rng.normal(loc=[0, 0], scale=0.3, size=(30, 2))
    blob_b = rng.normal(loc=[10, 10], scale=0.3, size=(30, 2))
    X = np.vstack([blob_a, blob_b])
    model = KMeansClustering(n_clusters=2).fit(X)
    assert model.labels_.shape == (60,)
    assert len(set(model.labels_)) == 2
    assert model.labels_[:30].tolist().count(model.labels_[0]) == 30
    assert model.labels_[30:].tolist().count(model.labels_[30]) == 30


def test_kmeans_predict_on_new_data():
    rng = np.random.default_rng(1)
    X = np.vstack([
        rng.normal(loc=[0, 0], scale=0.2, size=(20, 2)),
        rng.normal(loc=[8, 8], scale=0.2, size=(20, 2)),
    ])
    model = KMeansClustering(n_clusters=2).fit(X)
    new_pts = np.array([[0.0, 0.0], [8.0, 8.0]])
    preds = model.predict(new_pts)
    assert preds[0] != preds[1]


def test_kmeans_inertia_decreases_with_more_clusters():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(60, 2))
    inertia_2 = KMeansClustering(n_clusters=2).fit(X).inertia_
    inertia_5 = KMeansClustering(n_clusters=5).fit(X).inertia_
    assert inertia_5 < inertia_2


def test_kmeans_score_is_negative_inertia():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 2))
    model = KMeansClustering(n_clusters=3).fit(X)
    assert pytest.approx(model.score(X)) == -model.inertia_


def test_kmeans_predict_before_fit():
    model = KMeansClustering()
    with pytest.raises(AttributeError):
        model.predict(np.array([[1.0, 2.0]]))


def test_kmeans_score_before_fit():
    model = KMeansClustering()
    with pytest.raises(AttributeError):
        model.score(np.array([[1.0, 2.0]]))


def test_kmeans_empty_data():
    model = KMeansClustering()
    with pytest.raises(ValueError):
        model.fit(np.empty((0, 2)))


def test_kmeans_single_cluster():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(20, 3))
    model = KMeansClustering(n_clusters=1).fit(X)
    assert np.all(model.labels_ == 0)
    assert model.inertia_ >= 0
