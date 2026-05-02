import numpy as np
import pytest

from mlpackage.supervised_learning import MultilayerPerceptron


def test_mlp_learns_simple_classification():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(loc=-1.5, size=(40, 2)),
                    rng.normal(loc=1.5, size=(40, 2))])
    y = np.array([0] * 40 + [1] * 40)
    net = MultilayerPerceptron([2, 8, 2], activation="tanh", l2_penalty=0.001, rng_seed=1)
    net.fit(X, y, learning_rate=0.05, epochs=2000)
    acc = np.mean(net.predict(X) == y)
    assert acc >= 0.90


def test_mlp_multiclass():
    rng = np.random.default_rng(5)
    X = np.vstack([rng.normal(loc=[0, -2], size=(30, 2)),
                    rng.normal(loc=[2, 2], size=(30, 2)),
                    rng.normal(loc=[-2, 2], size=(30, 2))])
    y = np.array([0] * 30 + [1] * 30 + [2] * 30)
    net = MultilayerPerceptron([2, 16, 3], activation="relu", l2_penalty=0.001, rng_seed=3)
    net.fit(X, y, learning_rate=0.01, epochs=3000)
    preds = net.predict(X)
    assert preds.shape == (90,)
    assert set(np.unique(preds)).issubset({0, 1, 2})
    assert np.mean(preds == y) >= 0.80


def test_mlp_predict_probability_sums_to_one():
    rng = np.random.default_rng(10)
    X = rng.normal(size=(20, 3))
    y = rng.integers(0, 3, size=20)
    net = MultilayerPerceptron([3, 6, 3], rng_seed=2)
    net.fit(X, y, learning_rate=0.01, epochs=500)
    probs = net.predict_probability(X)
    assert probs.shape == (20, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all(probs >= 0.0)


def test_mlp_multiple_hidden_layers():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(50, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    net = MultilayerPerceptron([4, 8, 8, 2], activation="sigmoid", l2_penalty=0.0, rng_seed=0)
    net.fit(X, y, learning_rate=0.1, epochs=2000)
    assert net.predict(X).shape == (50,)


def test_mlp_empty_fit():
    net = MultilayerPerceptron([2, 4, 2])
    with pytest.raises(ValueError):
        net.fit(np.empty((0, 2)), np.array([]))


def test_mlp_shape_mismatch():
    net = MultilayerPerceptron([2, 4, 2])
    with pytest.raises(ValueError):
        net.fit(np.zeros((5, 2)), np.zeros(3, dtype=int))


def test_mlp_fit_returns_self():
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1])
    net = MultilayerPerceptron([2, 4, 2], rng_seed=0)
    assert net.fit(X, y, learning_rate=0.01, epochs=2) is net


def test_mlp_predict_returns_integer_class_labels():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 2))
    y = (X[:, 0] > 0).astype(int)
    net = MultilayerPerceptron([2, 4, 2], rng_seed=0)
    net.fit(X, y, learning_rate=0.05, epochs=100)
    preds = net.predict(X)
    assert np.issubdtype(preds.dtype, np.integer)
    assert preds.shape == (10,)


def test_mlp_predict_probability_rows_sum_to_one():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(15, 3))
    y = rng.integers(0, 2, size=15)
    net = MultilayerPerceptron([3, 5, 2], rng_seed=0)
    net.fit(X, y, learning_rate=0.01, epochs=100)
    probs = net.predict_probability(X)
    assert probs.shape == (15, 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-10)


def test_mlp_all_supported_activations_fit():
    """Ensure tanh, sigmoid, and relu each fit without error."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(25, 2))
    y = (X[:, 0] > 0).astype(int)
    for act in ("tanh", "sigmoid", "relu"):
        net = MultilayerPerceptron([2, 6, 2], activation=act, rng_seed=1)
        net.fit(X, y, learning_rate=0.05, epochs=100)
        preds = net.predict(X)
        assert preds.shape == (25,)


def test_mlp_invalid_activation_raises():
    """Using an unknown activation string must raise at forward time."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 2))
    y = np.array([0, 1, 0, 1, 0])
    net = MultilayerPerceptron([2, 4, 2], activation="foo", rng_seed=0)
    with pytest.raises(ValueError):
        net.fit(X, y, epochs=1)


def test_mlp_deterministic_with_same_seed():
    """Same seed, same data, same hyperparameters -> same predictions."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 2))
    y = (X[:, 0] > 0).astype(int)
    net1 = MultilayerPerceptron([2, 6, 2], rng_seed=7)
    net1.fit(X, y, learning_rate=0.05, epochs=200)
    net2 = MultilayerPerceptron([2, 6, 2], rng_seed=7)
    net2.fit(X, y, learning_rate=0.05, epochs=200)
    assert np.array_equal(net1.predict(X), net2.predict(X))


def test_mlp_deeper_network_trains():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(40, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    net = MultilayerPerceptron([3, 8, 8, 4, 2], activation="relu", rng_seed=1)
    net.fit(X, y, learning_rate=0.01, epochs=500)
    assert net.predict(X).shape == (40,)


def test_mlp_xor_problem():
    """A 2-hidden-layer network should solve XOR (not linearly separable)."""
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1, 1, 0])
    net = MultilayerPerceptron([2, 8, 2], activation="tanh", l2_penalty=0.0, rng_seed=2)
    net.fit(X, y, learning_rate=0.1, epochs=5000)
    assert np.array_equal(net.predict(X), y)


def test_mlp_prediction_matches_argmax_of_probabilities():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(12, 2))
    y = rng.integers(0, 3, size=12)
    net = MultilayerPerceptron([2, 6, 3], rng_seed=0)
    net.fit(X, y, learning_rate=0.01, epochs=50)
    probs = net.predict_probability(X)
    assert np.array_equal(np.argmax(probs, axis=1), net.predict(X))


def test_mlp_regularization_reduces_weight_norms():
    """Strong L2 regularization should lead to smaller weight norms than no regularization."""
    rng = np.random.default_rng(6)
    X = rng.normal(size=(40, 3))
    y = (X[:, 0] > 0).astype(int)
    net_strong = MultilayerPerceptron([3, 6, 2], l2_penalty=1.0, rng_seed=0)
    net_strong.fit(X, y, learning_rate=0.05, epochs=500)
    net_weak = MultilayerPerceptron([3, 6, 2], l2_penalty=0.0, rng_seed=0)
    net_weak.fit(X, y, learning_rate=0.05, epochs=500)
    strong_norm = np.linalg.norm(net_strong._W_final) + sum(
        np.linalg.norm(h.W) for h in net_strong._hidden
    )
    weak_norm = np.linalg.norm(net_weak._W_final) + sum(
        np.linalg.norm(h.W) for h in net_weak._hidden
    )
    assert strong_norm < weak_norm
