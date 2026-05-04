# Random Forest Regressor

Ensemble of **`DecisionTreeRegressor`** learners on bootstrap rows and random feature subsets; prediction is the **mean** of tree outputs. Implementation: [`src/mlpackage/supervised_learning/decision_tree_regressor.py`](../../../src/mlpackage/supervised_learning/decision_tree_regressor.py) (`RandomForestRegressor`).

**Mean aggregation formula, bagging variance intuition, link to CART base learners** are in the tutorial notebook.

## Hyperparameters

| Parameter      | Description |
| -------------- | ----------- |
| `n_estimators` | Number of trees. |
| `max_depth`    | Depth limit per tree. |
| `max_features` | `"sqrt"` or `None`. |

## Methods

`fit`, `predict`. Evaluate with **R²** and **RMSE** in your notebook (same spirit as `DecisionTreeRegressor.score` for R²).

## Practical notes

Set **`np.random.seed`** before `fit` for reproducibility; use train/test for honest metrics.

## Tutorial notebook

[`random_forest_regressor_tutorial.ipynb`](random_forest_regressor_tutorial.ipynb) — **Diabetes**, R², RMSE, residuals, truth vs predicted, **`n_estimators`** comparison.

`jupyter notebook examples/supervised_learning/random_forest_regressor/random_forest_regressor_tutorial.ipynb`

May save **`diabetes_random_forest_y_vs_yhat_scatter.png`**.
