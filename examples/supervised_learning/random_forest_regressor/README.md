# Random Forest Regressor

A **random forest regressor** averages predictions from many **bootstrap-trained** regression trees. Random column subsets per tree decorrelate errors so the ensemble often generalizes better than a single deep tree.

Implementation: [`src/mlpackage/supervised_learning/decision_tree_regressor.py`](../../../src/mlpackage/supervised_learning/decision_tree_regressor.py) (`RandomForestRegressor`).

## Training (this implementation)

For each of **`n_estimators`** trees:

1. Bootstrap sample rows (with replacement).
2. Subsample columns: **`max_features="sqrt"`** (default) or **`None`** for all features.
3. Fit `DecisionTreeRegressor(max_depth=...)` on the subsampled design.
4. Remember which column indices each tree uses.

## Prediction

The forest prediction is the **mean** of all tree predictions at each test row:

\\[
\\hat{y}(\\mathbf{x}) = \\frac{1}{B}\\sum_{b=1}^{B} T_b(\\mathbf{x}),
\\]

where \\(B\\) is **`n_estimators`**.

where each \\(T_b\\) is a **regression tree** fit on a bootstrap sample and a random feature subset (same mechanism as the classifier forest).

## Mathematical viewpoint (variance reduction)

### Bagging for regression

Let \\(Y_i\\) be noisy targets and consider idealized trees \\(T_b\\) trained on **bootstrap** replicates. The bagged regressor \\(\\bar{T}(\\mathbf{x}) = B^{-1}\\sum_b T_b(\\mathbf{x})\\) has variance

\\[
\\mathrm{Var}(\\bar{T}) = \\frac{1}{B^2}\\sum_{b,b'} \\mathrm{Cov}(T_b, T_{b'}) .
\\]

If trees were uncorrelated with common variance \\(\\sigma^2\\), \\(\\mathrm{Var}(\\bar{T}) = \\sigma^2/B\\). **Positive correlation** across trees (same data, similar structure) **inflates** ensemble variance but is still typically **below** the variance of any single high-variance tree—this is the **bias–variance tradeoff** motivating forests.

### Squared-error decomposition at a point

For squared error loss at \\(\\mathbf{x}\\), averaging predictions **cannot increase** the Bayes risk relative to choosing a single random tree when errors are negatively correlated enough; in practice, averaging smooths **jagged** tree surfaces into a **lower-variance** piecewise smoother estimate.

### Connection to CART base learners

Each \\(T_b\\) is the same **variance-reduction** greedy tree as `DecisionTreeRegressor`: piecewise constant on axis-aligned cells. The forest **averages** many such surfaces, approximating a **weighted ensemble** of local constant experts.

## Hyperparameters

| Parameter       | Description |
| --------------- | ----------- |
| `n_estimators`  | Number of trees. |
| `max_depth`     | Depth limit for each base tree. |
| `max_features`  | `"sqrt"` or `None`. |

## Methods

`fit`, `predict`. There is **no** `score` method—evaluate with **\(R^2\)** and **RMSE** in the notebook (same formulas as `DecisionTreeRegressor.score` for \(R^2\)).

## Practical notes

- Call **`np.random.seed(...)`** before `fit` for reproducibility.
- Use a **train/test split** for honest performance estimates.

## Tutorial notebook

[`random_forest_regressor_tutorial.ipynb`](random_forest_regressor_tutorial.ipynb)

From the repository root:

`jupyter notebook examples/supervised_learning/random_forest_regressor/random_forest_regressor_tutorial.ipynb`

Uses **Diabetes** (same spirit as the decision-tree regressor example): train/test split, forest fit, **\(R^2\)** and **RMSE**, residual preview, **truth vs predicted** scatter, and a comparison of **`n_estimators`**.

Step 7 saves **`diabetes_random_forest_y_vs_yhat_scatter.png`** next to the notebook when the working directory is that folder; from the repository root it writes under `examples/supervised_learning/random_forest_regressor/diabetes_random_forest_y_vs_yhat_scatter.png`.
