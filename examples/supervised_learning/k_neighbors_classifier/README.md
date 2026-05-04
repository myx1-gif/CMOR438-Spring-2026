# k-Nearest Neighbors Classifier

*k*-NN stores the training set and, at prediction time, finds the `n_neighbors` closest training points under **Euclidean distance**, then predicts by **majority vote** among their labels. Implementation: [`src/mlpackage/supervised_learning/knn.py`](../../../src/mlpackage/supervised_learning/knn.py).

Training is **`fit(X, y)`** (store arrays). There is **no** compact parametric model: cost scales with training set size. **Feature scaling** is important when variables use different units or scales.

**Distance formula, majority vote, Voronoi / local geometry intuition, and complexity** are in the tutorial notebook.

## Hyperparameters

| Parameter      | Type | Description |
| -------------- | ---- | ----------- |
| `n_neighbors`  | int  | Number of neighbors *k* (clipped to training size if larger). |

## Methods

- **`fit` / `predict` / `score`** — standard supervised API; `score` is mean accuracy.
- **`confusion_matrix`** — pandas cross-tab of true vs predicted.
- **`plot_decision_boundary`** — uses first two features only (2D grid).

## Practical notes

Brute-force O(n_train · p) per query in this educational code; curse of dimensionality in high *p*.

## Tutorial notebook

[`k_neighbors_classifier_tutorial.ipynb`](k_neighbors_classifier_tutorial.ipynb) — **Wine** data, stratified split, **StandardScaler** fit on train only, confusion matrix, 2D plot, **`n_neighbors`** comparison.

`jupyter notebook examples/supervised_learning/k_neighbors_classifier/k_neighbors_classifier_tutorial.ipynb`

Step 7 may save **`wine_two_features_scatter.png`**.
