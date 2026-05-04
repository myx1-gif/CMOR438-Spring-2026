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

## Mathematical foundation

At a node \\(N\\) containing targets \\(\\{y_i : i\\in N\\}\\) with \\(n = |N|\\), this code uses the **population** variance (divide by \\(n\\), `ddof=0`):

\\[
\\operatorname{Var}(N) = \\frac{1}{n}\\sum_{i\\in N}(y_i - \\bar{y}_N)^2, \\qquad
\\bar{y}_N = \\frac{1}{n}\\sum_{i\\in N} y_i .
\\]

### 1) Variance as impurity for regression trees

Unlike classification entropy, regression trees use **within-node spread** of \\(y\\) as impurity. Large \\(\\operatorname{Var}(N)\\) means responses in the cell vary widely; the split objective is to create children where \\(y\\) is more tightly clustered.

### 2) Variance reduction and its link to squared error

For a split into \\(L,R\\) with sizes \\(n_L,n_R\\), \\(n=n_L+n_R\\), define the **between-node pooled** variance

\\[
\\operatorname{Var}_{\\text{after}} = \\frac{n_L}{n}\\operatorname{Var}(L) + \\frac{n_R}{n}\\operatorname{Var}(R).
\\]

The implementation maximizes **variance drop**

\\[
\\Delta(j,t) = \\operatorname{Var}(N) - \\operatorname{Var}_{\\text{after}}.
\\]

For a fixed partition \\((L,R)\\), the **constant** \\(c_L,c_R\\) that minimize \\(\\sum_{i\\in L}(y_i-c_L)^2 + \\sum_{i\\in R}(y_i-c_R)^2\\) are \\(c_L=\\bar{y}_L\\), \\(c_R=\\bar{y}_R\\). Under those optimal constants, the **sum of squared errors** decomposes across the split in a way that makes **maximizing \\(\\Delta\\)** align with **greedy reduction of total SSE**—the same impurity philosophy as CART regression: each split greedily improves the best piecewise-constant approximation with one more breakpoint on an axis.

### 3) Leaf prediction as constant least squares

Each leaf predicts the **sample mean** of training \\(y\\) in that leaf:

\\[
\\hat{y}_L = \\frac{1}{|L|}\\sum_{i\\in L} y_i ,
\\]

which is the **L2-optimal** constant predictor for squared error on the training points assigned to \\(L\\).

### 4) \\(R^2\\) returned by `score`

Let \\(\\hat{y}_i\\) be tree predictions on a set of \\(n\\) pairs \\((\\mathbf{x}_i,y_i)\\), and \\(\\bar{y} = \\frac{1}{n}\\sum_i y_i\\). The implementation reports

\\[
R^2 = 1 - \\frac{\\sum_{i=1}^{n}(y_i-\\hat{y}_i)^2}{\\sum_{i=1}^{n}(y_i-\\bar{y})^2},
\\]

the usual **coefficient of determination** relative to the **mean baseline** on that same set. If \\(\\operatorname{Var}(y)=0\\) on the passed-in \\(y\\), the code returns `1.0`.

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

The notebook loads the **Diabetes** regression dataset from scikit-learn, performs a **train/test split**, fits `DecisionTreeRegressor` on the training fold only, reports **out-of-sample** \\(R^2\\), **MAE**, and **RMSE**, and includes a **true vs predicted** scatter plot (Step 7) plus a **`max_depth`** comparison table (Step 8).

Step 7 saves **`diabetes_y_vs_yhat_scatter.png`** in the same folder as the notebook if your working directory is that folder; if you start Jupyter from the repository root, it writes under `examples/supervised_learning/decision_tree_regressor/diabetes_y_vs_yhat_scatter.png`.
