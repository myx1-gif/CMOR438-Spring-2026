# Decision Tree Classifier

Decision trees are supervised models that recursively split the feature space with axis-aligned rules (`x_j <= t` vs `x_j > t`) until each region is mostly one class. This package’s `DecisionTreeClassifier` (in `src/mlpackage/supervised_learning/decision_tree_classifier.py`) is a binary, entropy-based tree: each split maximizes **information gain**, and leaves predict the **majority** label among training points in that leaf.

Training is **greedy** (each split is locally best, not globally optimal). **`max_depth`** limits how deep the tree can grow and is the main regularization knob. Trees are **interpretable** and do not require feature scaling for the split rules to be valid, though very deep trees can overfit.

**Detailed math** (entropy, information gain, leaf prediction, accuracy) lives in the tutorial notebook below, not in this README.

## Constructor parameters

| Parameter   | Type            | Description |
| ----------- | --------------- | ----------- |
| `max_depth` | `Optional[int]` | Maximum depth; `None` means grow until purity, depth cap, or no improving split. |

## Methods

- **`fit(X, y)`** — `X` shape `(n_samples, n_features)`, `y` shape `(n_samples,)`. Returns `self`. Raises if shapes are empty or mismatched.
- **`predict(X)`** — integer class labels per row. Requires `fit` first.
- **`score(X, y)`** — mean accuracy (fraction of matching labels).

## Practical notes

Interpretable paths, nonlinear boundaries, greedy splits, overfitting risk without depth control.

## Tutorial notebook

Install deps (`pip install -r requirements.txt`), then open [decision_tree_classifier_tutorial.ipynb](decision_tree_classifier_tutorial.ipynb) (step-by-step narrative and formulas) or run:

`jupyter notebook examples/supervised_learning/decision_tree_classifier/decision_tree_classifier_tutorial.ipynb`

Uses **Iris**, stratified train/test split, out-of-sample accuracy, optional 2D scatter, and a **`max_depth`** comparison. Step 7 may write **`iris_petal_scatter.png`** next to the notebook depending on your working directory.
