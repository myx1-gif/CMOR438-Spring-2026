"""Feedforward neural network (multi-layer perceptron) with softmax output."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def _apply_activation(z: np.ndarray, kind: str) -> np.ndarray:
    if kind == "tanh":
        return np.tanh(z)
    if kind == "sigmoid":
        z = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z))
    if kind == "relu":
        return np.maximum(0.0, z)
    raise ValueError(f"Unknown activation: {kind}")


def _activation_gradient(z: np.ndarray, kind: str) -> np.ndarray:
    if kind == "tanh":
        t = np.tanh(z)
        return 1.0 - t ** 2
    if kind == "sigmoid":
        s = _apply_activation(z, "sigmoid")
        return s * (1.0 - s)
    if kind == "relu":
        return (z > 0).astype(float)
    raise ValueError(f"Unknown activation: {kind}")


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / exp_vals.sum(axis=1, keepdims=True)


class DenseLayer:
    """Single fully-connected layer with an element-wise activation."""

    def __init__(
        self, n_inputs: int, n_units: int, activation: str = "tanh", rng_seed: int = 0
    ) -> None:
        self.n_inputs = n_inputs
        self.n_units = n_units
        self.activation = activation
        gen = np.random.RandomState(rng_seed)
        self.W = gen.randn(n_inputs, n_units) / np.sqrt(n_inputs)
        self.b = np.zeros((1, n_units))
        self.pre_activation: Optional[np.ndarray] = None
        self.output: Optional[np.ndarray] = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.pre_activation = X @ self.W + self.b
        self.output = _apply_activation(self.pre_activation, self.activation)
        return self.output


class MultilayerPerceptron:
    """
    Multi-layer perceptron classifier with softmax output trained via
    gradient descent and cross-entropy loss.

    Parameters
    ----------
    layer_sizes : list[int]
        Sizes of each layer including input and output.
        Example: ``[4, 16, 3]`` creates one hidden layer of 16 units,
        with 4 input features and 3 output classes.
    activation : str, default='tanh'
        Activation function for hidden layers ('tanh', 'sigmoid', 'relu').
    l2_penalty : float, default=0.01
        L2 regularization coefficient.
    rng_seed : int, default=0
        Base seed for weight initialization.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        activation: str = "tanh",
        l2_penalty: float = 0.01,
        rng_seed: int = 0,
    ) -> None:
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.l2_penalty = l2_penalty
        self._hidden: List[DenseLayer] = []

        for idx in range(len(layer_sizes) - 2):
            self._hidden.append(
                DenseLayer(
                    layer_sizes[idx],
                    layer_sizes[idx + 1],
                    activation,
                    rng_seed + idx,
                )
            )

        out_gen = np.random.RandomState(rng_seed)
        self._W_final = out_gen.randn(layer_sizes[-2], layer_sizes[-1]) / np.sqrt(
            layer_sizes[-2]
        )
        self._b_final = np.zeros((1, layer_sizes[-1]))
        self._class_probs: Optional[np.ndarray] = None

    def _forward(self, X: np.ndarray) -> np.ndarray:
        signal = X
        for layer in self._hidden:
            signal = layer.forward(signal)
        logits = signal @ self._W_final + self._b_final
        self._class_probs = _softmax(logits)
        return self._class_probs

    def _cross_entropy_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        n = X.shape[0]
        self._forward(X)
        log_likelihood = -np.sum(
            np.log(self._class_probs[np.arange(n), y] + 1e-15)
        ) / n
        reg = 0.5 * self.l2_penalty * (
            float(np.sum(self._W_final ** 2))
            + sum(float(np.sum(h.W ** 2)) for h in self._hidden)
        )
        return float(log_likelihood + reg)

    def _compute_gradients(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
        n = X.shape[0]
        self._forward(X)

        delta = self._class_probs.copy()
        delta[np.arange(n), y] -= 1.0

        last_hidden_out = self._hidden[-1].output if self._hidden else X
        grad_W_final = (last_hidden_out.T @ delta) / n
        grad_b_final = delta.sum(axis=0, keepdims=True) / n

        hidden_grads: List[Tuple[np.ndarray, np.ndarray]] = []
        propagated = delta

        for idx in reversed(range(len(self._hidden))):
            layer = self._hidden[idx]
            upstream_W = (
                self._W_final if idx == len(self._hidden) - 1 else self._hidden[idx + 1].W
            )
            propagated = (propagated @ upstream_W.T) * _activation_gradient(
                layer.pre_activation, layer.activation
            )
            input_to_layer = self._hidden[idx - 1].output if idx > 0 else X
            dW = (input_to_layer.T @ propagated) / n
            db = propagated.sum(axis=0, keepdims=True) / n
            hidden_grads.insert(0, (dW, db))

        return hidden_grads, grad_W_final, grad_b_final

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        learning_rate: float = 0.01,
        epochs: int = 20000,
    ) -> "MultilayerPerceptron":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        if X.size == 0 or y.size == 0:
            raise ValueError("Empty X or y provided.")
        if X.shape[0] != y.size:
            raise ValueError("Number of samples in X and y must match.")

        for _ in range(epochs):
            h_grads, gW_out, gb_out = self._compute_gradients(X, y)

            gW_out += self.l2_penalty * self._W_final
            self._W_final -= learning_rate * gW_out
            self._b_final -= learning_rate * gb_out

            for layer_idx, (gW, gb) in enumerate(h_grads):
                gW += self.l2_penalty * self._hidden[layer_idx].W
                self._hidden[layer_idx].W -= learning_rate * gW
                self._hidden[layer_idx].b -= learning_rate * gb

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._class_probs is None and not self._hidden:
            raise AttributeError("Model not fitted yet.")
        probs = self._forward(np.asarray(X, dtype=float))
        return np.argmax(probs, axis=1)

    def predict_probability(self, X: np.ndarray) -> np.ndarray:
        if self._class_probs is None and not self._hidden:
            raise AttributeError("Model not fitted yet.")
        return self._forward(np.asarray(X, dtype=float))
