# Multilayer Perceptron (MLP)

Feedforward network: one or more **dense hidden** layers with a chosen activation (`tanh`, `sigmoid`, `relu`), then a **linear** output and **softmax** for class probabilities. Training minimizes **multiclass cross-entropy** plus **L2 weight decay**, with **full-batch** gradient updates each epoch.

Implementation: [`src/mlpackage/supervised_learning/multilayer_perceptron.py`](../../../src/mlpackage/supervised_learning/multilayer_perceptron.py).

**`layer_sizes`** is `[input_dim, hidden..., output_dim]`. There is **no** `score` method—use `np.mean(model.predict(X) == y)` or your metric.

**Forward pass notation, softmax, loss + weight decay, backprop structure** are in the tutorial notebook.

## Hyperparameters

| Parameter       | Meaning |
| --------------- | ------- |
| `layer_sizes`   | Network shape including input and number of classes. |
| `activation`    | Hidden nonlinearity. |
| `l2_penalty`    | Weight decay coefficient. |
| `rng_seed`      | Initialization seed. |
| `learning_rate` | Passed into **`fit`**. |
| `epochs`        | Full-batch epochs in **`fit`**. |

## Practical notes

Scale inputs; tune learning rate and epochs; full-batch can be slow on large data (Iris-sized demos are fine).

## Tutorial notebook

[`multilayer_perceptron_tutorial.ipynb`](multilayer_perceptron_tutorial.ipynb) — **Iris**, scaling, accuracy, probabilities, 2D decision regions from a small 2-feature net, hidden-width comparison.

`jupyter notebook examples/supervised_learning/multilayer_perceptron/multilayer_perceptron_tutorial.ipynb`

May save **`iris_mlp_decision_regions.png`**.
