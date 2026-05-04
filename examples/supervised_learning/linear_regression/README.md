# Linear Regression

Linear regression models the response as an affine map of the inputs,

\\[
y \\approx \\theta_0 + \\sum_{j=1}^{p}\\theta_j x_j ,
\\]

with parameters \\(\\theta = (\\theta_0, \\theta_1, \\ldots, \\theta_p)\\). Estimating \\(\\theta\\) from data is the canonical **supervised regression** problem.

This package's `LinearRegression` (see [`src/mlpackage/supervised_learning/linear_regression.py`](../../../src/mlpackage/supervised_learning/linear_regression.py)) fits **ordinary least squares (OLS)** by solving the **normal equations** with the **Moore–Penrose pseudoinverse**:

\\[
\\boldsymbol{\\theta} = (\\mathbf{X}^\\top \\mathbf{X})^{+} \\mathbf{X}^\\top \\mathbf{y},
\\]

where \\(\\mathbf{X}\\) is the **design matrix**: a column of ones for the **intercept** concatenated with your feature columns.

## Optimization objective

OLS minimizes the residual sum of squares:

\\[
\\min_{\\boldsymbol{\\theta}} \\; \\|\\mathbf{y} - \\mathbf{X}\\boldsymbol{\\theta}\\|_2^2 .
\\]

Using `numpy.linalg.pinv` handles rank-deficient or ill-conditioned feature matrices more robustly than a literal matrix inverse.

## Learned parameters

After `fit`, access:

| Attribute    | Description                                                                        |
| ------------ | ---------------------------------------------------------------------------------- |
| `intercept`  | Scalar bias \\(\\theta_0\\).                                                       |
| `coef_`      | 1D array of slopes \\((\\theta_1,\\ldots,\\theta_p)\\), shape `(n_features,)`. |

Predictions: \\(\\hat{\\mathbf{y}} = \\mathbf{X}_{\\text{nofeat}} \\boldsymbol{\\theta}_{\\text{coef}} + \\theta_0\\) implemented as `X @ coef_ + intercept`.

## Methods

### `fit(X, y)`

- **`X`:** `(n_samples, n_features)` numeric features (no intercept column; it is added internally).
- **`y`:** `(n_samples,)` continuous targets.

### `predict(X)`

- Returns `(n_samples,)` predicted \\(\\hat{y}\\).

### `rmse(X, y)`

\\[
\\mathrm{RMSE} = \\sqrt{\\frac{1}{n}\\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2}
\\]

### `R_squared(X, y)` (coefficient of determination)

Let \\(\\bar{y} = \\frac{1}{n}\\sum_i y_i\\) computed from the **`y` passed in**.

\\[
R^2 = 1 - \\frac{\\sum_{i}(y_i - \\hat{y}_i)^2}{\\sum_{i}(y_i - \\bar{y})^2}
\\]

If all \\(y_i\\) are identical, this implementation returns `1.0`.

There is **no** sklearn-style `score` alias on this class—use **`R_squared`** and **`rmse`** explicitly.

## Practical notes

- **Interpretability:** Coefficients describe additive contributions when features are measured on comparable scales (often standardized for interpretation).
- **Linearity:** Captures global linear trends; nonlinear relationships may need transformed features or other models.
- **Outliers:** Squared loss is sensitive to large residuals.

## Tutorial notebook (Jupyter)

Open [`linear_regression_tutorial.ipynb`](linear_regression_tutorial.ipynb) or run:

`jupyter notebook examples/supervised_learning/linear_regression/linear_regression_tutorial.ipynb`

The notebook uses the **Diabetes** regression dataset (same spirit as the decision tree regressor example): train/test split, fit `LinearRegression`, report **`R_squared`** and **`rmse`** on train and test, inspect residuals and coefficients, plot **held-out truth vs predicted**, and compare against a **constant baseline** that always predicts **`mean(y_train)`**.

Step 7 saves **`diabetes_linear_y_vs_yhat_scatter.png`** next to the notebook when the working directory is that folder; from the repo root it writes under `examples/supervised_learning/linear_regression/diabetes_linear_y_vs_yhat_scatter.png`.
