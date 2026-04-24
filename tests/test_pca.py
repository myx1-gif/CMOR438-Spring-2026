import numpy as np
import pytest

from mlpackage.unsupervised_learning import PrincipalComponentAnalysis


def test_fit_transform_reduces_dimensions():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 5))
    pca = PrincipalComponentAnalysis(n_components=2)
    X_reduced = pca.fit_transform(X)
    assert X_reduced.shape == (50, 2)


def test_all_components_when_none():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 4))
    pca = PrincipalComponentAnalysis(n_components=None).fit(X)
    assert pca.axes_.shape == (4, 4)
    assert pca.eigenvalues_.shape == (4,)


def test_variance_ratio_sums_to_one_all_components():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 3))
    pca = PrincipalComponentAnalysis().fit(X)
    assert pytest.approx(pca.variance_ratio_.sum(), abs=1e-10) == 1.0


def test_eigenvalues_sorted_descending():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(60, 6))
    pca = PrincipalComponentAnalysis().fit(X)
    for i in range(len(pca.eigenvalues_) - 1):
        assert pca.eigenvalues_[i] >= pca.eigenvalues_[i + 1]


def test_transform_before_fit_raises():
    pca = PrincipalComponentAnalysis()
    with pytest.raises(AttributeError):
        pca.transform(np.array([[1.0, 2.0]]))


def test_empty_data_raises():
    pca = PrincipalComponentAnalysis()
    with pytest.raises(ValueError):
        pca.fit(np.empty((0, 3)))


def test_reconstruction_error_small():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(30, 3))
    pca = PrincipalComponentAnalysis(n_components=3).fit(X)
    X_proj = pca.transform(X)
    X_reconstructed = X_proj @ pca.axes_ + pca.feature_mean_
    assert np.allclose(X, X_reconstructed, atol=1e-10)


def test_single_feature():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(20, 1))
    pca = PrincipalComponentAnalysis(n_components=1).fit(X)
    assert pca.axes_.shape == (1, 1)
    X_t = pca.transform(X)
    assert X_t.shape == (20, 1)
