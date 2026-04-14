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
├── pytest.ini
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
│       │   └── decision_tree_regressor.py
│       └── unsupervised_learning
│           └── __init__.py
└── tests
    ├── test_decision_tree.py
    ├── test_decision_tree_regressor.py
    └── test_smoke.py
```

---

## Implemented Algorithms

### Supervised Learning
- **Decision Tree Classifier** (entropy + information gain)
- **Random Forest Classifier** (bootstrap aggregation + feature subsampling)
- **Decision Tree Regressor** (variance reduction)
- **Random Forest Regressor** (bootstrap aggregation + feature subsampling)

## Utilities

- `mlpackage.preprocess`
- `mlpackage.metrics`

---

## Testing

The test suite currently covers:
- classifier and regressor decision tree behavior
- random forest shape/reproducibility checks
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
