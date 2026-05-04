# Logistic Regression

Logistic regression models the probability that \\(y=1\\) with a **sigmoid** of a linear score:

\\[
P(y=1 \\mid \\mathbf{x}) = \\sigma(\\mathbf{w}^\\top \\mathbf{x} + b), \\qquad
\\sigma(z) = \\frac{1}{1 + e^{-z}} .
\\]

This implementation in [`src/mlpackage/supervised_learning/logistic_regression.py`](../../../src/mlpackage/supervised_learning/logistic_regression.py) is **binary** only: targets should behave like \\(\\{0,1\\}\\). Training minimizes **binary cross-entropy** using **batch gradient descent**.

## Loss and gradients

With \\(p_i = \\sigma(\\mathbf{w}^\\top \\mathbf{x}_i + b)\\),

\\[
\\mathcal{L} = -\\frac{1}{n}\\sum_{i=1}^{n} \\Bigl[ y_i \\log p_i + (1-y_i)\\log(1-p_i) \\Bigr].
\\]

Gradients (average over samples):

\\[
\\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{w}} = \\frac{1}{n}\\mathbf{X}^\\top(\\mathbf{p}-\\mathbf{y}), \\qquad
\\frac{\\partial \\mathcal{L}}{\\partial b} = \\frac{1}{n}\\sum_i (p_i - y_i).
\\]

Updates each iteration:

\\[
\\mathbf{w} \\leftarrow \\mathbf{w} - \\eta \\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{w}}, \\qquad
b \\leftarrow b - \\eta \\frac{\\partial \\mathcal{L}}{\\partial b}.
\\]

\\(\\eta\\) is **`learning_rate`**; **`n_iterations`** controls how many full-batch steps run.

`numpy.clip` inside `_sigmoid` limits \\(z\\) for numerical stability.

## Hyperparameters

| Parameter          | Type   | Description                                      |
| ------------------ | ------ | ------------------------------------------------ |
| `learning_rate`    | `float`| Gradient step size \\(\\eta\\).                  |
| `n_iterations`     | `int`  | Number of batch gradient descent iterations.      |

## Attributes after `fit`

| Attribute | Description                                           |
| --------- | ----------------------------------------------------- |
| `weights` | Shape `(n_features,)`, corresponds to \\(\\mathbf{w}\\). |
| `bias`    | Scalar \\(b\\).                                       |

## Methods

### `fit(X, y)`

- **`X`:** `(n_samples, n_features)` floats.
- **`y`:** `(n_samples,)` — training treats **`y` as floats** in \\(\\{0,1\\}\\).

### `predict_probability(X)`

Returns \\(P(y=1\\mid X)\\) per row in \\((0,1)\\).

### `predict(X)`

Class rule: **1** if \\(P(y=1\\mid x) \\ge 0.5\\), else **0** (returns integer array).

### `score(X, y)`

Mean **accuracy** (fraction of exact class matches).

## Practical notes

- **Feature scaling** (e.g. `StandardScaler` fit on training data) often improves optimization for gradient descent and makes `learning_rate` easier to choose.
- This is **not** regularized (no L1/L2 term in the loss as implemented).
- For **multiclass** problems, use a different model or reframe the data (e.g. one-vs-rest outside this class).

## Tutorial notebook (Jupyter)

Open [`logistic_regression_tutorial.ipynb`](logistic_regression_tutorial.ipynb) or run:

`jupyter notebook examples/supervised_learning/logistic_regression/logistic_regression_tutorial.ipynb`

The notebook uses the **Wisconsin Breast Cancer** dataset (two classes, 30 numeric features). It uses a **stratified** train/test split, **standardizes** features using training statistics, fits `LogisticRegression`, reports **accuracy** and a small **confusion-style** view, visualizes two scaled features, and **compares** a few **`n_iterations`** values on the same split.

Step 7 saves **`breast_cancer_two_features_scatter.png`** next to the notebook when the working directory is that folder; from the repository root it writes under `examples/supervised_learning/logistic_regression/breast_cancer_two_features_scatter.png`.
