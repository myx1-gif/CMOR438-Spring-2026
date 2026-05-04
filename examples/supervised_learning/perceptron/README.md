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

### Geometric view (linear separator)

The decision boundary is the hyperplane \\(\\{\\mathbf{x} : \\mathbf{w}^\\top \\mathbf{x} + b = 0\\}\\). The **signed distance** of a point to that hyperplane (up to normalization \\(\\|\\mathbf{w}\\|\\)) is proportional to \\(\\mathbf{w}^\\top \\mathbf{x} + b\\). The perceptron update moves \\(\\mathbf{w}\\) by \\(\\eta (y_i-\\hat{y}_i)\\mathbf{x}_i\\): when \\(y_i=1\\) but \\(\\hat{y}_i=0\\), \\(\\mathbf{w}\\) rotates toward \\(\\mathbf{x}_i\\); when \\(y_i=0\\) but \\(\\hat{y}_i=1\\), it rotates away. Thus mistakes drive all parameter change.

### Mistake bound (separable case, classical)

If there exists some \\((\\mathbf{w}^\\star, b^\\star)\\) that **strictly separates** the two classes with margin \\(\\gamma>0\\) on the training set and \\(\\|\\mathbf{x}_i\\|\\le R\\), Rosenblatt’s perceptron algorithm (variants of the online rule) makes at most \\((R/\\gamma)^2\\) mistakes up to scaling—explaining why **finite** mistakes occur under **linear separability**, while overlapping classes can cause **non-convergence** or cycling unless one adds averaging or margins.

### Relation to hinge loss and logistic regression

The perceptron criterion can be viewed as related to **non-differentiable** surrogates of classification error; **hinge SVM** uses a margin-based hinge, while **logistic regression** uses a smooth log-loss. The hard threshold here yields **sparse** updates (only when \\(\\hat{y}_i \\neq y_i\\) in the discrete sense used by the implementation’s update \\((y_i-\\hat{y}_i)\\in\\{-1,0,1\\}\\)).

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
