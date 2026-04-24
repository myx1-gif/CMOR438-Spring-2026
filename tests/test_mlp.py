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
