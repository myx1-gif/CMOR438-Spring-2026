# Decision Tree Classifier

Decision trees are supervised learning models used for classification (and in other variants, regression). They work by recursively partitioning the feature space into regions that are increasingly homogeneous in class label.

This package's `DecisionTreeClassifier` (implemented in `src/mlpackage/supervised_learning/decision_tree_classifier.py`) is a binary, entropy-based classifier that chooses splits by maximizing information gain.

## Algorithm Overview

The model builds a tree of binary decision rules of the form:

- if `x_j <= t`, go to left child
- otherwise, go to right child

where `j` is a feature index and `t` is a threshold.

Training is greedy and recursive:

1. Start with all training samples at the root node.
2. For each feature, evaluate candidate thresholds from observed feature values.
3. Compute information gain for each candidate split.
4. Choose the split with the highest gain.
5. Recurse on left/right subsets until a stopping condition is met.
6. Assign a class label to each leaf.

During inference, each sample traverses the tree from root to leaf, and the leaf class is returned.

## Mathematical Foundation

Let a node contain labels  y = y_1, \dots, y_n , and let class proportions be:


p_k = \frac{i : y_i = k}{n}


### 1) Node impurity via Shannon entropy


H(y) = -\sum_{k} p_k \log_2(p_k)


- H(y)=0 when the node is pure (all one class).
- H(y) is larger when classes are more mixed.

### 2) Information gain for a split

Given a candidate split that partitions labels into `left` and `right`:


\text{IG} = H(\text{parent}) - \left(\frac{n_L}{n}H(\text{left}) + \frac{n_R}{n}H(\text{right})\right)


where n_L and n_R are the number of samples in left/right child.

The selected split is:


(j^*, t^*) = \arg\max_{j,t} \text{IG}(j,t)


This implementation computes candidate thresholds from unique observed values in each feature and chooses the best valid binary split.

### 3) Leaf prediction rule

When splitting stops, a node predicts:


\hat{y}_{leaf} = \operatorname{mode}(y)


If a node is already pure, that class is returned directly.

## Stopping Conditions in This Implementation

Recursion stops when any of the following is true:

1. Node is pure (`labels.size == 1`).
2. Maximum depth is reached (`depth >= max_depth`, if provided).
3. No valid split improves partitioning (no usable feature/threshold pair).

## Parameters, Inputs, and Outputs

## Constructor Parameters


| Parameter   | Type            | Description                                                                                                                             |
| ----------- | --------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `max_depth` | `Optional[int]` | Maximum tree depth. If `None`, tree grows until purity/no valid split. Smaller values regularize complexity and can reduce overfitting. |


## `fit(X, y)`

- **Input `X`:** 2D numeric array of shape `(n_samples, n_features)`.
- **Input `y`:** 1D label array of shape `(n_samples,)`.
- **Output:** returns fitted `DecisionTreeClassifier` instance (`self`).
- **Validation:** raises `ValueError` if `X`/`y` are empty or sample counts do not match.

## `predict(X)`

- **Input `X`:** 2D numeric array of samples to classify.
- **Output:** 1D NumPy array of predicted integer class labels.
- **Error case:** raises `AttributeError` if called before `fit`.

## `score(X, y)`

- **Input:** feature matrix `X` and true labels `y`.
- **Output:** scalar float accuracy:


\text{accuracy} = \frac{1}{n}\sum_{i=1}^{n}\mathbf{1}\hat{y}_i = y_i


## Practical Characteristics

- **Interpretable:** predictions are explainable as rule paths.
- **Nonlinear:** captures nonlinear decision boundaries.
- **Low preprocessing burden:** no feature scaling required for split correctness.
- **Overfitting risk:** deep trees can memorize data; `max_depth` helps regularize.
- **Greedy optimization:** each split is locally optimal, not globally optimal.

## Tutorial notebook (Jupyter)

From the repository root, install dependencies if needed (`pip install -r requirements.txt`), then start Jupyter and open:

[decision_tree_classifier_tutorial.ipynb](decision_tree_classifier_tutorial.ipynb)

Alternatively from a terminal:

`jupyter notebook examples/supervised_learning/decision_tree_classifier/decision_tree_classifier_tutorial.ipynb`

The notebook loads **Iris** via scikit-learn, performs a stratified **train/test split**, fits `DecisionTreeClassifier` on the training fold only, and reports **out-of-sample** predictions and accuracy on the held-out test fold.

Step 7 (scatter plot) also saves **`iris_petal_scatter.png`** next to this README (same folder as the notebook) if your working directory is that folder; if you launch Jupyter from the repository root instead, it writes under `examples/supervised_learning/decision_tree_classifier/iris_petal_scatter.png`.