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

## Mathematical foundation

Consider a node \\(N\\) containing \\(n\\) labeled samples \\((\\mathbf{x}_i, y_i)\\) with class labels in a finite set \\(\\mathcal{K}\\). Let

\\[
n_k = \\sum_{i\\in N} \\mathbf{1}\\{y_i = k\\}, \\qquad p_k = \\frac{n_k}{n}
\\]

be class counts and empirical class proportions at that node.

### 1) Shannon entropy as impurity

The implementation uses **binary entropy** (log base 2):

\\[
H(N) = -\\sum_{k\\in\\mathcal{K}} p_k \\log_2 p_k ,
\\]

with the convention \\(0\\log_2 0 = 0\\). Entropy is **concave** in \\(\\mathbf{p}\\): it is zero iff one \\(p_k=1\\) (pure node), and maximized at the uniform distribution over classes present (most “mixed” for fixed support).

### 2) Information gain and conditional entropy

A candidate axis-aligned split on feature \\(j\\) at threshold \\(t\\) partitions \\(N\\) into a left child \\(L\\) (points with \\(x_{ij}\\le t\\)) and right child \\(R\\) (strict inequality in code: \\(x_{ij} > t\\)). Define the **weighted child entropy**

\\[
H_{\\text{after}}(j,t) = \\frac{n_L}{n} H(L) + \\frac{n_R}{n} H(R).
\\]

**Information gain** is the drop in entropy:

\\[
\\mathrm{IG}(j,t) = H(N) - H_{\\text{after}}(j,t).
\\]

Equivalently, \\(\\mathrm{IG}\\) is the **mutual information** \\(I(Y;\\, \\mathbf{1}\\{X_j \\le t\\})\\) under the empirical distribution at the node: it measures how much knowing which side of the split a point falls on reduces uncertainty about \\(Y\\).

The learner chooses

\\[
(j^\\*, t^\\*) \\in \\arg\\max_{j,t\\in\\mathcal{T}_j} \\mathrm{IG}(j,t),
\\]

where \\(\\mathcal{T}_j\\) is the set of **distinct observed** thresholds for feature \\(j\\) on the training points in the node (this is the standard exhaustive search for univariate CART-style splits on numeric data).

### 3) Greedy tree growing as a piecewise-constant classifier

Inductively, each split refines a **partition** of \\(\\mathbb{R}^p\\) into axis-aligned cells; every leaf cell predicts a **constant** class label (majority vote). Thus the fitted classifier is a **piecewise-constant** function of \\(\\mathbf{x}\\). Training is **greedy**: each split maximizes \\(\\mathrm{IG}\\) **locally** at one node; this does **not** solve the global problem of finding the smallest-error tree (that problem is combinatorially hard). Depth control (`max_depth`) limits how many refinements are allowed, trading bias and variance.

### 4) Leaf prediction

At a leaf \\(L\\) with labels \\(\\{y_i : i\\in L\\}\\), the implementation predicts

\\[
\\hat{y}_L = \\arg\\max_{k} \\sum_{i\\in L} \\mathbf{1}\\{y_i = k\\},
\\]

the **majority class** (mode). If the node is already pure, this equals the common label of all points in the leaf.

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
- **Output:** scalar float **0–1 accuracy**:

\\[
\\mathrm{Acc} = \\frac{1}{n}\\sum_{i=1}^{n} \\mathbf{1}\\{\\hat{y}_i = y_i\\}.
\\]


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