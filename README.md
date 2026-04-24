# MLPackage: Data Science and Machine Learning Project

## Overview
`MLPackage` is a course project package focused on clean implementations of core machine learning algorithms, tests, and reproducible examples.

It includes a Python package in `src/mlpackage`, unit tests in `tests`, and notebook examples in `examples`.

---

## Project Structure

```bash
.
├── .github
│   └── workflows
│       └── ci.yml
├── data
│   ├── processed
│   └── raw
├── examples
│   └── package_demo.ipynb
├── pyproject.toml
├── README.md
├── requirements.txt
├── setup.py
├── src
│   └── mlpackage
│       ├── __init__.py
│       ├── metrics.py
│       ├── preprocess.py
│       ├── supervised_learning
│       │   ├── __init__.py
│       │   ├── decision_tree_classifier.py
│       │   ├── decision_tree_regressor.py
│       │   ├── knn.py
│       │   ├── linear_regression.py
│       │   ├── logistic_regression.py
│       │   ├── multilayer_perceptron.py
│       │   └── perceptron.py
│       └── unsupervised_learning
│           ├── __init__.py
│           ├── dbscan.py
│           ├── kmeans.py
│           ├── label_propagation.py
│           └── pca.py
└── tests
    ├── test_decision_tree.py
    ├── test_decision_tree_regressor.py
    ├── test_knn.py
    ├── test_linear_regression.py
    ├── test_logistic_regression.py
    ├── test_mlp.py
    ├── test_perceptron.py
    ├── test_dbscan.py
    ├── test_kmeans.py
    ├── test_label_propagation.py
    ├── test_pca.py
    └── test_smoke.py
```

---

## Implemented Algorithms

### Supervised Learning
- **Decision Tree Classifier** (entropy + information gain)
- **Random Forest Classifier** (bootstrap aggregation + feature subsampling)
- **Decision Tree Regressor** (variance reduction)
- **Random Forest Regressor** (bootstrap aggregation + feature subsampling)
- **K-Nearest Neighbors Classifier** (Euclidean distance, majority vote; confusion matrix and 2D boundary plot)
- **Linear Regression** (normal equation with pseudoinverse; RMSE and R²)
- **Logistic Regression** (batch gradient descent on binary cross-entropy; probability and class output)
- **Multi-Layer Perceptron** (feedforward neural network with softmax output, backpropagation, L2 regularization)
- **Perceptron** (single-layer binary classifier with step activation and online weight updates)

### Unsupervised Learning
- **DBSCAN** (density-based clustering with noise detection)
- **K-Means** (iterative centroid refinement with inertia tracking)
- **Label Propagation** (graph-based semi-supervised learning with RBF similarity and clamped propagation)
- **PCA** (principal component analysis via covariance eigen-decomposition for dimensionality reduction)

## Utilities

- `mlpackage.preprocess`
- `mlpackage.metrics`

---

## Testing

The test suite currently covers:
- classifier and regressor decision tree behavior
- random forest shape/reproducibility checks
- KNN and linear regression
- input validation and unfitted-model errors
- basic package import smoke test

Run tests with:

```bash
pytest
```

---

## Installation

```bash
git clone <repo_url>
cd <repo_folder>
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

---

## Quick Start

```python
import numpy as np
from mlpackage.supervised_learning import DecisionTreeClassifier

X = np.array([[0], [1], [2], [3]])
y = np.array([0, 0, 1, 1])

model = DecisionTreeClassifier(max_depth=2)
model.fit(X, y)
predictions = model.predict(np.array([[0.5], [2.5]]))
print(predictions)
```

---

## License

See `LICENSE`.
