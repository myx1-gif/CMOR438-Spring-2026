# k-Nearest Neighbors Classifier

*k*-nearest neighbors (KNN) is a **memory-based** supervised classifier: there is no iterative optimization during training beyond storing the training set. At prediction time, each query compares feature vectors to stored examples using a distance and aggregates neighbor labels.

This package's `KNeighborsClassifier` lives in [`src/mlpackage/supervised_learning/knn.py`](../../../src/mlpackage/supervised_learning/knn.py). It uses **Euclidean distance** and **majority voting** among the `n_neighbors` closest training labels.

## Algorithm overview

For each test row \\(\\mathbf{x}\\):

1. Compute Euclidean distance to every training row \\(\\mathbf{x}^{(i)}\\).
2. Sort training indices by ascending distance.
3. Take the **`n_neighbors`** smallest distances (with \\(k\\) clipped to training size if `n_neighbors` exceeds \\(n_{\\mathrm{train}}\\)).
4. Predict the **mode** (majority class) among those neighbor labels using label counts.

Training **`fit(X, y)`** simply stores converted arrays (lazy learning).

## Mathematical foundation

### Euclidean distance

Between \\(\\mathbf{x}, \\mathbf{z} \\in \\mathbb{R}^p\\),

\\[
d(\\mathbf{x}, \\mathbf{z}) = \\sqrt{\\sum_{j=1}^{p}(x_j - z_j)^2}.
\\]

Because distance sums squared coordinate differences, **feature scale matters**: a coordinate with large numeric range can dominate unless features are comparable (often via scaling).

### Majority vote

Let neighbor labels be \\(y_{(1)},\\ldots,y_{(k)}\\). This implementation predicts

\\[
\\hat{y} = \\arg\\max_{c} \\sum_{\\ell=1}^{k} \\mathbf{1}\\{ y_{(\\ell)} = c \\},
\\]

implemented via `numpy.bincount` over integer class ids.

### Decision regions and the Bayes classifier (intuition)

With **\\(k=1\\)**, the classifier is a **nearest-neighbor** rule: \\(\\hat{y}(\\mathbf{x}) = y_{\\pi(1)}\\) where \\(\\pi(1)\\) is the index of the closest training point. In \\(\\mathbb{R}^p\\), the set of points closer to training point \\(\\mathbf{x}^{(i)}\\) than to any other training point forms a **Voronoi cell**; 1-NN induces a **Voronoi partition** of space labeled by training tags.

For **general \\(k\\)**, the prediction is a **local majority vote** within the metric ball ordering induced by distances: as \\(k\\) grows, predictions **smooth** (higher bias, lower variance under common asymptotics); small \\(k\\) tracks the training labels more closely (low bias, high variance). In the **two-class** case with odd \\(k\\), ties in the vote count are impossible; for multiclass ties, `argmax` of counts picks a deterministic winner among tied classes.

### Metric structure (this implementation)

Only the **Euclidean** \\(\\ell_2\\) metric is implemented. More generally \\(\\ell_q\\) distances \\(d(\\mathbf{x},\\mathbf{z}) = (\\sum_j |x_j-z_j|^q)^{1/q}\\) weight coordinate differences differently; **Mahalanobis** distance \\(\\sqrt{(\\mathbf{x}-\\mathbf{z})^\\top \\mathbf{\\Sigma}^{-1}(\\mathbf{x}-\\mathbf{z})}\\) rescales correlated features. Euclidean KNN is **equivariant to rigid motions** (translations/rotations) of feature space after scaling, but **not** invariant to arbitrary linear transforms unless you bake the metric accordingly—hence **per-feature standardization** when units differ.

### Complexity

A single query costs **\\(O(n_{\\mathrm{train}} \\cdot p)\\)** time and **\\(O(n_{\\mathrm{train}})\\)** memory to store data; there is no compact parametric model. Fast \\(k\\)-NN libraries use spatial data structures (KD-trees, locality-sensitive hashing); this educational code uses brute-force loops.

## Hyperparameters

| Parameter       | Type  | Description                                                                 |
| --------------- | ----- | --------------------------------------------------------------------------- |
| `n_neighbors` | `int` | Number of neighbors \\(k\\). Typical odd values (e.g. 3, 5, 7) reduce ties in **binary** problems; ties in multiclass are resolved by `bincount.argmax` behavior. |

## Methods and I/O

### `fit(X, y)`

- **`X`:** shape `(n_samples, n_features)`, cast to `float`.
- **`y`:** shape `(n_samples,)`, integer class labels.
- **Returns:** `self`.

### `predict(X)`

- **`X`:** query rows, shape `(n_queries, n_features)`.
- **Returns:** integer predictions per row.

### `score(X, y)` / `accuracy(X, y)`

- **Returns:** mean accuracy \\(\\frac{1}{n}\\sum_i \\mathbf{1}\\{\\hat{y}_i = y_i\\}\\).

### `confusion_matrix(X, y)`

- **Returns:** `pandas.DataFrame` with rows = true class, columns = predicted class (counts).

### `plot_decision_boundary(X, y)` / `draw_decision_boundary`

- Expects **at least two** numeric columns; only the **first two** are used to build a 2D grid and contour plot. Useful only for small 2D slices or projections.

## Practical notes

- **Scaling:** Strongly recommended for Euclidean KNN when features differ in units or spread (see tutorial: **fit scaler on training data only**).
- **Cost:** Distance to all training points per query is \\(O(n_{\\mathrm{train}} \\cdot p)\\) per test row with this implementation (exact brute force).
- **Curse of dimensionality:** distance concentration can hurt in very high dimensions without careful representation learning.

## Tutorial notebook (Jupyter)

Open [`k_neighbors_classifier_tutorial.ipynb`](k_neighbors_classifier_tutorial.ipynb) or run:

`jupyter notebook examples/supervised_learning/k_neighbors_classifier/k_neighbors_classifier_tutorial.ipynb`

The notebook uses the **Wine** dataset from scikit-learn (three cultivar classes, 13 continuous chemical features). Feature scales differ across columns, which makes it a good illustration of **standardization before KNN**. It performs a stratified train/test split, fits `StandardScaler` on the training fold only, evaluates accuracy and a confusion matrix, plots two scaled features, and compares several **`n_neighbors`** values on the **same** split.

Step 7 saves **`wine_two_features_scatter.png`** next to the notebook when the working directory is that folder; from the repository root it writes under `examples/supervised_learning/k_neighbors_classifier/wine_two_features_scatter.png`.
