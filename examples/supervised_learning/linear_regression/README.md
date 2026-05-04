# Linear Regression

This package’s `LinearRegression` ([`src/mlpackage/supervised_learning/linear_regression.py`](../../../src/mlpackage/supervised_learning/linear_regression.py)) fits **ordinary least squares**: an intercept plus feature weights chosen to minimize squared residuals. The implementation solves via the **normal equations** using **`numpy.linalg.pinv`** (Moore–Penrose inverse), which behaves better than a raw matrix inverse when features are rank-deficient or ill-conditioned.

After **`fit`**, use **`intercept`** and **`coef_`**. There is **no** sklearn-style `score`—use **`R_squared`** and **`rmse`** explicitly.

**OLS objective, normal equations, projection view, hat matrix / leverage sketch** are in the tutorial notebook.

## Methods

- **`fit(X, y)`** — `X` is `(n_samples, n_features)`; intercept column is added internally.
- **`predict(X)`** — affine predictions.
- **`rmse(X, y)`**, **`R_squared(X, y)`** — metrics on the pair you pass (train or test).

## Practical notes

Globally linear; squared loss is sensitive to outliers; coefficient interpretation is easier on comparable scales (often standardized features).

## Tutorial notebook

[`linear_regression_tutorial.ipynb`](linear_regression_tutorial.ipynb) — **Diabetes**, train/test, coefficients, residuals, truth vs predicted, comparison to a **constant mean** baseline.

`jupyter notebook examples/supervised_learning/linear_regression/linear_regression_tutorial.ipynb`

Step 7 may save **`diabetes_linear_y_vs_yhat_scatter.png`**.
