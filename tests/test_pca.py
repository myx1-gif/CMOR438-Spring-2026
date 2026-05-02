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


def test_fit_returns_self():
    X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    pca = PrincipalComponentAnalysis()
    assert pca.fit(X) is pca


def test_feature_mean_equals_column_mean():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(15, 3))
    pca = PrincipalComponentAnalysis().fit(X)
    assert np.allclose(pca.feature_mean_, X.mean(axis=0), atol=1e-12)


def test_axes_are_orthonormal():
    """Principal axes should form an orthonormal basis (rows)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(25, 4))
    pca = PrincipalComponentAnalysis().fit(X)
    product = pca.axes_ @ pca.axes_.T
    assert np.allclose(product, np.eye(pca.axes_.shape[0]), atol=1e-10)


def test_known_2d_principal_axis_direction():
    """Data along y = x should have a principal axis aligned with (1, 1)/sqrt(2)."""
    X = np.array([[i, i] for i in range(20)], dtype=float)
    pca = PrincipalComponentAnalysis(n_components=1).fit(X)
    axis = pca.axes_[0]
    axis = axis / np.linalg.norm(axis)
    expected = np.array([1.0, 1.0]) / np.sqrt(2)
    assert np.isclose(abs(np.dot(axis, expected)), 1.0, atol=1e-10)


def test_transform_output_shape():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 6))
    pca = PrincipalComponentAnalysis(n_components=3).fit(X)
    X_t = pca.transform(rng.normal(size=(9, 6)))
    assert X_t.shape == (9, 3)


def test_projection_has_zero_mean_on_training_data():
    """Projected training data should have approximately zero column means."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 3))
    pca = PrincipalComponentAnalysis(n_components=2).fit(X)
    X_proj = pca.transform(X)
    assert np.allclose(X_proj.mean(axis=0), 0.0, atol=1e-10)


def test_variance_ratio_entries_nonnegative_and_sorted():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 4))
    pca = PrincipalComponentAnalysis().fit(X)
    assert np.all(pca.variance_ratio_ >= -1e-12)
    for i in range(len(pca.variance_ratio_) - 1):
        assert pca.variance_ratio_[i] + 1e-10 >= pca.variance_ratio_[i + 1]


def test_projected_variances_match_eigenvalues():
    """Column variance of the projection should equal the corresponding eigenvalue
    (within normalization; np.cov uses n-1 denominator)."""
    rng = np.random.default_rng(4)
    X = rng.normal(size=(100, 5))
    pca = PrincipalComponentAnalysis(n_components=5).fit(X)
    X_proj = pca.transform(X)
    proj_var = X_proj.var(axis=0, ddof=1)
    assert np.allclose(proj_var, pca.eigenvalues_, atol=1e-10)


def test_n_components_greater_than_features_capped():
    """Requesting more components than features should not error but may cap to d."""
    rng = np.random.default_rng(5)
    X = rng.normal(size=(10, 3))
    pca = PrincipalComponentAnalysis(n_components=10).fit(X)
    assert pca.axes_.shape[0] <= 3


def test_pca_defaults():
    pca = PrincipalComponentAnalysis()
    assert pca.n_components is None
    assert pca.axes_ is None
    assert pca.eigenvalues_ is None
    assert pca.variance_ratio_ is None
    assert pca.feature_mean_ is None


def test_pca_fit_transform_equivalent_to_fit_then_transform():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(20, 4))
    pca_a = PrincipalComponentAnalysis(n_components=2).fit(X)
    A = pca_a.transform(X)
    pca_b = PrincipalComponentAnalysis(n_components=2)
    B = pca_b.fit_transform(X)
    assert np.allclose(A, B, atol=1e-12)
