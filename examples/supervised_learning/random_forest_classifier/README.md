# Random Forest Classifier

An **ensemble** of `DecisionTreeClassifier` models, each fit on a **bootstrap** sample of rows and a random subset of features (`max_features="sqrt"` by default, or all features if `None`). Predictions are **majority vote** across trees.

Defined next to `DecisionTreeClassifier` in [`src/mlpackage/supervised_learning/decision_tree_classifier.py`](../../../src/mlpackage/supervised_learning/decision_tree_classifier.py). Set **`numpy.random.seed`** before **`fit`** if you need reproducible bagging and feature subsampling.

**Voting formula, bagging / OOB intuition, random subspaces, correlation vs idealized vote** are in the tutorial notebook.

## Hyperparameters

| Parameter      | Description |
| -------------- | ----------- |
| `n_estimators` | Number of trees. |
| `max_depth`    | Depth limit per base tree. |
| `max_features` | `"sqrt"` or `None`. |

## Methods

`fit`, `predict`. No `score` on this class—use `np.mean(model.predict(X) == y)`.

## Practical notes

No OOB error in this educational code; use a held-out test set.

## Tutorial notebook

[`random_forest_classifier_tutorial.ipynb`](random_forest_classifier_tutorial.ipynb) — **Iris**, accuracy, 2D regions from a two-feature forest, **`n_estimators`** comparison.

`jupyter notebook examples/supervised_learning/random_forest_classifier/random_forest_classifier_tutorial.ipynb`

May save **`iris_random_forest_decision_regions.png`**.
