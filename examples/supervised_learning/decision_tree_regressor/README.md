# Decision Tree Regressor

Regression is supervised learning where the target **y** is numeric (counts, temperatures, concentrations, progression scores). A model learns a mapping \\(f:\\mathbb{R}^p\\to \\mathbb{R}\\) so that \\(f(\\mathbf{x})\\) approximates the observed \\(y\\) on training data.

This package's `DecisionTreeRegressor` (implemented in [`src/mlpackage/supervised_learning/decision_tree_regressor.py`](../../../src/mlpackage/supervised_learning/decision_tree_regressor.py)) is an educational binary tree that chooses splits **using variance reduction** (weighted decrease in variance of \\(y\\) after a split).

## Algorithm Overview

Splits have the binary form:

- if \\(x_j \\le t\\), go to left child  
- otherwise, go to right child  

Training is greedy and recursive:

1. Start with all training samples at the root.
2. For each feature, try candidate thresholds from observed feature values.
3. Score each split by how much it **reduces weighted variance** of \\(y\\) in the children.
4. Choose the split with the largest reduction.
5. Recurse until a stopping condition applies.
6. At a leaf, predict the **mean** of \\(y\\) in that node's training subset.

Inference routes each \\(\\mathbf{x}\\) root-to-leaf and returns that leaf mean.

## Mathematical Foundation

Suppose a node holds targets \\(y_1,\\ldots,y_n\\) with population-style variance  
\\(\\operatorname{Var}(y)=\\frac{1}{n}\\sum_{i=1}^{n}(y_i-\\bar y)^2\\) (`numpy.var(..., ddof=0)`, matching this code).

### 1) Impurity / node homogeneity

Larger variance means \\(y\\) is more spread out inside the region; splitting aims to concentrate similar responses in each child.

### 2) Variance reduction from a candidate split

Let a split partition indices into left (\\(n_L\\) points) and right (\\(n_R\\) points), \\(n=n_L+n_R\\). Define **weighted child variance**:

\\[
\\operatorname{Var}_{\\text{split}} = \\frac{n_L}{n}\\operatorname{Var}(y^{(L)}) + \\frac{n_R}{n}\\operatorname{Var}(y^{(R)})
\\]

This implementation maximizes:

\\[
\\Delta = \\operatorname{Var}(y^{\\text{parent}}) - \\operatorname{Var}_{\\text{split}}
\\]

(or equivalently **minimizes** \\(\\operatorname{Var}_{\\text{split}}\\) after fixing the parent). The chosen \\((j^*, t^*)\\) is the pair with largest \\(\\Delta\\) among splits that assign at least one point to each side.

### 3) Leaf prediction within this implementation

For any leaf node \\(L\\),

\\[
\\hat{y}_{\\mathrm{leaf}} = \\frac{1}{|L|}\\sum_{i\\in L} y_i .
\\]

## Stopping Conditions in This Implementation

Recursion stops when any of these holds:

1. **Constant targets at the node** (`np.unique(y).size == 1`): predicts that value (equal to its mean).
2. **`max_depth` reached**, if specified.
3. **No usable split**: no `(feature_index, threshold)` produces two non-empty children with positive variance reduction (the implementation returns a leaf with the **mean** of \\(y\\) at that node).

## Parameters, Inputs, and Outputs

### Constructor parameters

| Parameter   | Type            | Description                                                                                                                             |
| ----------- | --------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `max_depth` | `Optional[int]` | Maximum tree depth. If `None`, the tree grows until a stopping rule above triggers. Smaller depth limits capacity and can reduce overfitting. |

### `fit(X, y)`

- **Input `X`:** 2D numeric array, shape `(n_samples, n_features)`.
- **Input `y`:** 1D numeric targets, shape `(n_samples,)`.
- **Output:** returns the fitted `DecisionTreeRegressor` (`self`).
- **Errors:** `ValueError` if `X`/`y` empty or row counts disagree.

### `predict(X)`

- **Input `X`:** 2D numeric array of rows to predict.
- **Output:** 1D `float` array of predicted values.
- **Error:** `AttributeError` if used before `fit`.

### `score(X, y)` (coefficient of determination, \\(R^2\\))

Let \\(\\hat{y}_i\\) be predictions and \\(y_i\\) true values, \\(\\bar y = \\frac{1}{n}\\sum_i y_i\\).

\\[
R^2 = 1 - \\frac{\\sum_{i=1}^{n}(y_i-\\hat{y}_i)^2}{\\sum_{i=1}^{n}(y_i-\\bar{y})^2}
\\]

- **Edge case:** if all \\(y_i\\) are equal (denominator 0), this implementation returns `1.0`.

Higher \\(R^2\\) is better on the data you pass in; for **honest** generalization, evaluate on a **held-out** test set that was not used in `fit`.

## Practical characteristics

- **Interpretable** rule paths; **nonlinear** relationships without feature scaling for split rules.
- **Greedy** splits are not globally optimal.
- **Very deep** trees can overfit; tune `max_depth`.
- For interpretability of error in original units, also report **MAE** or **RMSE** alongside \\(R^2\\) (see tutorial notebook).

## Tutorial notebook (Jupyter)

From the repository root, install dependencies if needed (`pip install -r requirements.txt`), then open:

[decision_tree_regressor_tutorial.ipynb](decision_tree_regressor_tutorial.ipynb)

Or from a terminal:

`jupyter notebook examples/supervised_learning/decision_tree_regressor/decision_tree_regressor_tutorial.ipynb`

The notebook loads the **Diabetes** regression dataset from scikit-learn, performs a **train/test split**, fits `DecisionTreeRegressor` on the training fold only, reports **out-of-sample** \\(R^2\\), **MAE**, and **RMSE**, and includes an optional **true vs predicted** scatter plot.

When you run the optional plot cell, it saves **`diabetes_y_vs_yhat_scatter.png`** in the same folder as the notebook if your working directory is that folder; if you start Jupyter from the repository root, it writes under `examples/supervised_learning/decision_tree_regressor/diabetes_y_vs_yhat_scatter.png`.
