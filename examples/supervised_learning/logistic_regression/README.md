# Logistic Regression

Binary classifier in [`src/mlpackage/supervised_learning/logistic_regression.py`](../../../src/mlpackage/supervised_learning/logistic_regression.py). It models the probability of class 1 with a **sigmoid** of a linear score and trains by **batch gradient descent** on **binary cross-entropy**. Targets should behave like **0/1**.

Hyperparameters: **`learning_rate`**, **`n_iterations`**. After **`fit`**, **`weights`** and **`bias`** hold the linear part. **`predict_probability`** returns calibrated-ish probabilities; **`predict`** thresholds at 0.5. This build is **not** L1/L2 regularized in the loss.

**Loss, gradients, Bernoulli likelihood, logit link, convexity sketch, relation to perceptron** are in the tutorial notebook.

## Hyperparameters

| Parameter       | Type  | Description |
| --------------- | ----- | ----------- |
| `learning_rate` | float | Gradient step size. |
| `n_iterations`  | int   | Number of full-batch updates. |

## Methods

`fit`, `predict_probability`, `predict`, `score` (accuracy).

## Practical notes

Feature scaling (e.g. `StandardScaler` on train) usually helps gradient descent; multiclass needs another framing outside this class.

## Tutorial notebook

[`logistic_regression_tutorial.ipynb`](logistic_regression_tutorial.ipynb) — **Breast Cancer**, stratified split, scaling, accuracy, iteration comparison, 2D decision boundary figure.

`jupyter notebook examples/supervised_learning/logistic_regression/logistic_regression_tutorial.ipynb`

May save **`breast_cancer_logistic_decision_boundary.png`**.
