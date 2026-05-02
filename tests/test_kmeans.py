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


def test_kmeans_fit_returns_self():
    X = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0], [6.0, 6.0]])
    model = KMeansClustering(n_clusters=2)
    assert model.fit(X) is model


def test_kmeans_centers_shape():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4))
    model = KMeansClustering(n_clusters=3).fit(X)
    assert model.centers_.shape == (3, 4)


def test_kmeans_centers_are_means_of_assigned_points():
    """After convergence, each centroid should equal the mean of its cluster members."""
    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.normal(loc=[0, 0], scale=0.1, size=(20, 2)),
        rng.normal(loc=[10, 10], scale=0.1, size=(20, 2)),
    ])
    model = KMeansClustering(n_clusters=2, max_steps=200).fit(X)
    for k in range(2):
        members = X[model.labels_ == k]
        assert np.allclose(model.centers_[k], members.mean(axis=0), atol=1e-8)


def test_kmeans_inertia_zero_when_k_equals_n_with_unique_points():
    """If each point is its own cluster, inertia should be ~0."""
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    model = KMeansClustering(n_clusters=4, max_steps=10).fit(X)
    assert np.isclose(model.inertia_, 0.0, atol=1e-10)


def test_kmeans_labels_are_integers_in_valid_range():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(40, 2))
    k = 4
    model = KMeansClustering(n_clusters=k).fit(X)
    assert np.all(model.labels_ >= 0)
    assert np.all(model.labels_ < k)
    assert np.issubdtype(model.labels_.dtype, np.integer)


def test_kmeans_deterministic_with_fixed_seed():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 2))
    model_a = KMeansClustering(n_clusters=3).fit(X, seed=123)
    model_b = KMeansClustering(n_clusters=3).fit(X, seed=123)
    assert np.array_equal(model_a.labels_, model_b.labels_)
    assert np.allclose(model_a.centers_, model_b.centers_)


def test_kmeans_different_seed_may_differ():
    """Different seeds can yield different initial centers; check it runs OK."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 2))
    m1 = KMeansClustering(n_clusters=3).fit(X, seed=1)
    m2 = KMeansClustering(n_clusters=3).fit(X, seed=2)
    assert m1.labels_.shape == m2.labels_.shape


def test_kmeans_predict_on_same_training_data_matches_labels_():
    """Predicting on training X should return the stored labels."""
    rng = np.random.default_rng(2)
    X = np.vstack([
        rng.normal(loc=[0, 0], scale=0.3, size=(10, 2)),
        rng.normal(loc=[8, 8], scale=0.3, size=(10, 2)),
    ])
    model = KMeansClustering(n_clusters=2).fit(X)
    assert np.array_equal(model.predict(X), model.labels_)


def test_kmeans_inertia_matches_sum_of_squared_distances():
    """Stored inertia should equal manual SSE computation."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 2))
    model = KMeansClustering(n_clusters=3).fit(X)
    manual = 0.0
    for k in range(3):
        members = X[model.labels_ == k]
        manual += float(np.sum((members - model.centers_[k]) ** 2))
    assert np.isclose(model.inertia_, manual, atol=1e-8)


def test_kmeans_defaults():
    model = KMeansClustering()
    assert model.n_clusters == 3
    assert model.max_steps == 100
    assert model.convergence_tol == 1e-4
    assert model.centers_ is None
    assert model.labels_ is None
    assert model.inertia_ is None


def test_kmeans_predict_shape():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(25, 2))
    model = KMeansClustering(n_clusters=2).fit(X)
    preds = model.predict(rng.normal(size=(5, 2)))
    assert preds.shape == (5,)
