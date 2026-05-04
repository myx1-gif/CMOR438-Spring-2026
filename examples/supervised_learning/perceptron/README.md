# Perceptron

The **single-layer perceptron** here is a **binary** linear classifier: prediction applies a **hard threshold** to a linear score \\( \\mathbf{w}^\\top \\mathbf{x} + b \\). Training uses Rosenblatt-style **online** updates cycling through samples each epoch.

Implementation: [`src/mlpackage/supervised_learning/perceptron.py`](../../../src/mlpackage/supervised_learning/perceptron.py).

## Model

For sample \\(\\mathbf{x}\\),

\\[
\\hat{y} = \\mathbb{1}\\{ \\mathbf{w}^\\top \\mathbf{x} + b \\ge 0 \\},
\\]

with outputs interpreted as **0** or **1** (targets must use the same encoding).

## Update rule

For each training sample \\((\\mathbf{x}_i, y_i)\\) with \\(y_i \\in \\{0,1\\}\\), compare \\(\\hat{y}_i\\) to \\(y_i\\) and apply:

\\[
\\mathbf{w} \\leftarrow \\mathbf{w} + \\eta (y_i - \\hat{y}_i)\\, \\mathbf{x}_i, \\qquad
b \\leftarrow b + \\eta (y_i - \\hat{y}_i),
\\]

with **`lr`** = \\(\\eta\\). One **epoch** loops over all samples once; **`max_iter`** is the number of epochs.

After each epoch the implementation records **mean squared error** \\(\\frac{1}{n}\\sum_i (y_i - \\hat{y}_i)^2\\) on the **training** set in **`training_errors`**.

## Hyperparameters

| Parameter   | Meaning |
| ----------- | ------- |
| `lr`        | Learning rate \\(\\eta\\) for weight updates. |
| `max_iter`  | Number of epochs (full passes over training rows). |

## Methods

### `fit(X, y)`

- **`y`** must be compatible with binary **0/1** targets.

### `predict(X)`

Returns **0/1** labels via the step activation.

### `score(X, y)`

Accuracy (fraction of matching labels).

### `confusion_matrix(X, y)`

Returns a **pandas** cross-tab (actual vs predicted).

### `plot_training_loss()`

Convenience matplotlib plot of **`training_errors`** (calls **`plt.show()`** internally).

## Practical notes

- Data should often be **scaled** so feature magnitudes are comparable (`StandardScaler` fit on training data).
- Linear separation is **not** guaranteed; the perceptron may cycle or plateau when data are not separable.
- Unlike softmax logistic regression, there are **no calibrated probabilities**—only hard labels.

## Tutorial notebook (Jupyter)

Open [`perceptron_tutorial.ipynb`](perceptron_tutorial.ipynb) or:

`jupyter notebook examples/supervised_learning/perceptron/perceptron_tutorial.ipynb`

Uses **Wisconsin Breast Cancer** (binary), stratified split, standardized features, **`Perceptron`**, accuracy + confusion matrix, **MSE-vs-epoch curve** from **`training_errors`**, a **2D decision boundary** plot from a **two-feature** refit, and a **`max_iter`** comparison table.

Figures saved:

- **`breast_cancer_perceptron_training_mse.png`** — training MSE after each epoch  
- **`breast_cancer_perceptron_decision_boundary.png`** — shaded regions and test points (same folder as the notebook if cwd matches; otherwise under `examples/supervised_learning/perceptron/` from repo root)
