# Multilayer Perceptron (MLP)

A **multilayer perceptron** here is a feedforward network: one or more **dense hidden** layers with a fixed nonlinearity, then a **linear** readout followed by **softmax** for multiclass probabilities. Training minimizes **cross-entropy** plus **L2 weight decay**, using **full-batch gradient descent** each epoch.

Implementation: [`src/mlpackage/supervised_learning/multilayer_perceptron.py`](../../../src/mlpackage/supervised_learning/multilayer_perceptron.py).

## Architecture

`layer_sizes` is a list **[input_dim, hidden₁, …, output_dim]**:

- Example **`[4, 16, 3]`**: 4 input features, one hidden layer of 16 units, 3 output classes.
- Hidden layers use the same **`activation`** (`"tanh"`, `"sigmoid"`, or `"relu"`).
- The output layer has **no** separate activation before softmax; logits \\(\\mathbf{z}\\) feed:

\\[
P(y=k \\mid \\mathbf{x}) = \\frac{e^{z_k}}{\\sum_j e^{z_j}} .
\\]

## Forward pass (notation)

Let `layer_sizes` be \\([d_0, d_1, \\ldots, d_L]\\) with input \\(\\mathbf{a}^{(0)} = \\mathbf{x} \\in \\mathbb{R}^{d_0}\\). For each hidden layer \\(\\ell=1,\\ldots,L-1\\),

\\[
\\mathbf{z}^{(\\ell)} = \\mathbf{W}^{(\\ell)} \\mathbf{a}^{(\\ell-1)} + \\mathbf{b}^{(\\ell)}, \\qquad
\\mathbf{a}^{(\\ell)} = \\phi\\big(\\mathbf{z}^{(\\ell)}\\big),
\\]

where \\(\\phi\\) is `tanh`, `sigmoid`, or `relu` applied elementwise. Logits at the output layer are \\(\\mathbf{z}^{(L)}\\); class probabilities are

\\[
p_k = \\frac{e^{z^{(L)}_k}}{\\sum_j e^{z^{(L)}_j}} .
\\]

## Loss (cross-entropy + weight decay)

For integer labels \\(y_i\\in\\{0,\\ldots,K-1\\}\\) and softmax outputs \\(\\mathbf{p}_i\\), the **average multiclass cross-entropy** is

\\[
\\mathcal{L}_{\\mathrm{CE}} = -\\frac{1}{n}\\sum_{i=1}^{n} \\log p_{i,y_i}.
\\]

The implementation adds **L2 regularization** on all weight matrices (hidden \\(\\mathbf{W}^{(\\ell)}\\) and final output weights \\(\\mathbf{W}^{(L)}\\)):

\\[
\\mathcal{L} = \\mathcal{L}_{\\mathrm{CE}} + \\frac{\\lambda}{2}\\Big(\\|\\mathbf{W}^{(L)}\\|_F^2 + \\sum_{\\ell=1}^{L-1} \\|\\mathbf{W}^{(\\ell)}\\|_F^2\\Big),
\\]

with **`l2_penalty`** = \\(\\lambda\\). Gradients on weights pick up an extra **\\(\\lambda \\mathbf{W}\\)** term (Tikhonov / ridge on weights), shrinking parameters toward zero to reduce overfitting.

### Backpropagation (structure)

Let \\(\\delta^{(L)} = \\mathbf{p} - \\mathbf{y}_{\\text{one-hot}}\\) for the softmax layer. Propagating upstream,

\\[
\\delta^{(\\ell)} = \\big((\\mathbf{W}^{(\\ell+1)})^\\top \\delta^{(\\ell+1)}\\big) \\odot \\phi'(\\mathbf{z}^{(\\ell)}),
\\]

and weight gradients are outer products of activations and downstream signals (averaged over the batch in this code). This is the **multivariate chain rule** applied systematically along the computation graph.

## Hyperparameters

| Parameter       | Meaning |
| --------------- | ------- |
| `layer_sizes`   | Network shape including input and number of classes. |
| `activation`    | Hidden nonlinearity: `"tanh"`, `"sigmoid"`, `"relu"`. |
| `l2_penalty`    | Coefficient \\(\\lambda\\) for weight decay. |
| `rng_seed`      | Seeds random initialization. |
| `learning_rate` | Step size, passed to **`fit`**. |
| `epochs`        | Number of full-batch gradient updates in **`fit`**. |

## Methods

### `fit(X, y, learning_rate=0.01, epochs=20000)`

- **`X`:** `(n_samples, n_features)` floats.
- **`y`:** integer class indices `0 … n_classes-1`.

### `predict(X)`

Returns the **argmax** class per row.

### `predict_probability(X)`

Returns the **softmax** matrix, shape `(n_samples, n_classes)`.

There is **no** `score` method—use **`np.mean(model.predict(X) == y)`** (or your own metric).

## Practical notes

- **Scale inputs** (e.g. `StandardScaler` fit on training data) so gradient steps behave reasonably.
- **Epoch count** and **learning rate** strongly affect convergence; this educational implementation does **not** log loss per epoch inside `fit`.
- Full-batch updates on large data can be slow; Iris-sized problems are fine for demos.

## Tutorial notebook (Jupyter)

Open [`multilayer_perceptron_tutorial.ipynb`](multilayer_perceptron_tutorial.ipynb) or:

`jupyter notebook examples/supervised_learning/multilayer_perceptron/multilayer_perceptron_tutorial.ipynb`

The notebook uses **Iris** (4 features, 3 classes): stratified split, **standardized** features, trains an MLP on **all** features, evaluates accuracy and class probabilities, plots **2D decision regions** using a **smaller MLP fit on two features only** (for visualization), and compares a few **hidden widths** on the same split.

Step 7 saves **`iris_mlp_decision_regions.png`** next to the notebook when the working directory is that folder; from the repository root it writes under `examples/supervised_learning/multilayer_perceptron/iris_mlp_decision_regions.png`.
