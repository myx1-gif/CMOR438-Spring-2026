# Perceptron

Binary **linear** classifier with a **hard threshold** on a linear score, trained with **Rosenblatt-style online updates** over epochs. Implementation: [`src/mlpackage/supervised_learning/perceptron.py`](../../../src/mlpackage/supervised_learning/perceptron.py).

Hyperparameters: **`lr`**, **`max_iter`**. **`training_errors`** records training MSE after each epoch (for the step curve). **`plot_training_loss`** plots it.

**Update rule, geometric separator view, separability / mistake-bound intuition, contrast with hinge/logistic** are in the tutorial notebook.

## Methods

`fit`, `predict`, `score`, `confusion_matrix`, `plot_training_loss`.

## Practical notes

Scale features; no probabilities; may not converge if classes are not linearly separable.

## Tutorial notebook

[`perceptron_tutorial.ipynb`](perceptron_tutorial.ipynb) — **Breast Cancer**, scaling, accuracy, confusion matrix, MSE curve, 2D boundary, **`max_iter`** table.

`jupyter notebook examples/supervised_learning/perceptron/perceptron_tutorial.ipynb`

May save **`breast_cancer_perceptron_training_mse.png`** and **`breast_cancer_perceptron_decision_boundary.png`**.
