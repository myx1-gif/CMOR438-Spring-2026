# Random Forest Classifier

A **random forest** is an **ensemble** of decision trees trained on **bootstrap** resamples of the data. Predictions are aggregated—here by **majority vote** across trees—so individual tree variance is averaged out while nonlinear structure is retained.

This package’s `RandomForestClassifier` is defined in [`src/mlpackage/supervised_learning/decision_tree_classifier.py`](../../../src/mlpackage/supervised_learning/decision_tree_classifier.py) next to `DecisionTreeClassifier`. Each base learner is an entropy-splitting tree; the forest injects randomness through **row resampling with replacement** and optional **column subsampling**.

## How training works (this implementation)

For each of **`n_estimators`** trees:

1. Draw **`n_samples`** row indices **with replacement** (a bootstrap sample).
2. Choose a random subset of feature columns:
   - **`max_features="sqrt"`** (default): `max(1, floor(sqrt(n_features)))` columns, chosen uniformly without replacement.
   - **`max_features=None`**: use all features.
3. Fit a `DecisionTreeClassifier(max_depth=...)` on the bootstrapped rows and selected columns only.
4. Store the tree together with the **column index list** so predictions map back to the full feature vector.

## Prediction

For a test row \\(\\mathbf{x}\\), each tree outputs a class label. The forest returns the **mode** (majority vote) over trees. Ties are resolved by `numpy.bincount(...).argmax()` (smallest class index among tied maxima).

## Mathematical viewpoint (bagging + random subspaces)

Let \\(T_b\\) denote the \\(b\\)-th randomized tree predictor (bootstrap sample + random feature subset). The forest classifier is

\\[
\\hat{y}(\\mathbf{x}) = \\arg\\max_{k} \\sum_{b=1}^{B} \\mathbf{1}\\{ T_b(\\mathbf{x}) = k \\},
\\]

where \\(B\\) is the constructor argument **`n_estimators`**.

### Bootstrap variance (intuition)

Each \\(T_b\\) is trained on an i.i.d. **with-replacement** sample of size \\(n\\) from the empirical distribution; the **out-of-bag** fraction is about \\((1-1/n)^n \\approx e^{-1}\\), leaving roughly **63%** in-bag per tree on average. **Bagging** averages (or votes over) **conditionally independent** (given data) randomized predictors to **reduce variance** when the base learner is **unstable** (small changes in data change predictions a lot—deep trees).

Formally, for squared error regression analogs, the variance of the average of \\(B\\) identically distributed correlated predictors decomposes into a **\\(1/B\\)** term plus a correlation penalty; **random feature subsets** decorrelate trees so the ensemble benefits more from averaging than \\(B\\) copies of the same tree would.

### Random subspace method

Choosing a random subset \\(\\mathcal{S}_b \\subseteq \\{1,\\ldots,p\\}\\) of features for each tree injects **diversity**: different trees probe different projections of \\(\\mathbb{R}^p\\), reducing covariance between their errors on test points. Using **`max_features="sqrt"`** uses about \\(\\sqrt{p}\\) features per split search per tree, a common heuristic in high dimensions.

### Majority vote as Bayes under tree independence (rough intuition)

If trees were **independent** classifiers with equal error rates, majority vote corresponds to reducing the probability that **more than half** are wrong—binomial tail probabilities shrink exponentially in \\(B\\). Real trees are correlated, so the benefit is less than the idealized analysis, but the same mechanism drives ensemble gains.

## Hyperparameters

| Parameter       | Description |
| --------------- | ----------- |
| `n_estimators`  | Number of trees in the ensemble. More trees usually stabilize predictions at higher compute cost. |
| `max_depth`     | Depth cap passed to each `DecisionTreeClassifier` (`None` = grow until stopping rules in the tree). |
| `max_features`  | `"sqrt"` or `None` — controls random feature subset size per tree. |

## Methods

### `fit(X, y)`

Stores an ensemble of fitted trees.

### `predict(X)`

Returns integer class labels (majority vote). There is **no** `score` method on this class—compute accuracy with `np.mean(model.predict(X) == y)`.

## Practical notes

- **Set `numpy.random.seed`** before `fit` if you need reproducible forests (bootstrap and feature draws use the global RNG).
- Trees do **not** require feature scaling for split rules, but scaling can still matter if you compare to distance-based models on the same notebook.
- Out-of-bag error is **not** implemented here—use a held-out test set.

## Tutorial notebook

[`random_forest_classifier_tutorial.ipynb`](random_forest_classifier_tutorial.ipynb)

From the repository root:

`jupyter notebook examples/supervised_learning/random_forest_classifier/random_forest_classifier_tutorial.ipynb`

Uses **Iris** (stratified split), fits a forest on all four features, evaluates accuracy, plots **2D decision regions** from a **two-feature** refit (petal length & width) for visualization, and compares **`n_estimators`**.

Step 7 saves **`iris_random_forest_decision_regions.png`** next to the notebook when the working directory is that folder; from the repository root it writes under `examples/supervised_learning/random_forest_classifier/iris_random_forest_decision_regions.png`.
