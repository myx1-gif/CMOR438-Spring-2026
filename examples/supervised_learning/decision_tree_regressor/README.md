# Decision Tree Regressor

Regression predicts a **numeric** target from features. This package’s `DecisionTreeRegressor` ([`src/mlpackage/supervised_learning/decision_tree_regressor.py`](../../../src/mlpackage/supervised_learning/decision_tree_regressor.py)) builds a binary tree that chooses splits to **reduce weighted variance** of the target in the child nodes; each leaf predicts the **mean** of training targets in that leaf.

Same story as classification trees: **greedy** growth, **`max_depth`** for capacity control, **no scaling required** for threshold rules. **`score(X, y)`** returns **R²** (coefficient of determination vs the mean baseline on the data you pass in).

**Variance formulas, variance reduction, R², and leaf means** are derived in the tutorial notebook.

## Constructor parameters

| Parameter   | Type            | Description |
| ----------- | --------------- | ----------- |
| `max_depth` | `Optional[int]` | Depth cap; `None` until stopping rules trigger. |

## Methods

- **`fit(X, y)`** — continuous `y`, 2D `X`. Returns `self`.
- **`predict(X)`** — float predictions (leaf means).

Stopping rules in code include constant targets at a node, depth limit, or no split that improves variance reduction.

## Practical notes

Interpretable, can overfit if very deep; report MAE/RMSE alongside R² for interpretability in original units.

## Tutorial notebook

[decision_tree_regressor_tutorial.ipynb](decision_tree_regressor_tutorial.ipynb) — **Diabetes**, train/test split, out-of-sample R², MAE, RMSE, truth vs predicted plot, **`max_depth`** table.

`jupyter notebook examples/supervised_learning/decision_tree_regressor/decision_tree_regressor_tutorial.ipynb`

Step 7 may save **`diabetes_y_vs_yhat_scatter.png`** next to the notebook (path depends on cwd).
