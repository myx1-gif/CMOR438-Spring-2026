# Unsupervised Learning

Unsupervised learning is the branch of machine learning concerned with discovering structure in data that has no labeled outcome to predict. Instead of learning a mapping from inputs to known targets, an unsupervised model is asked to describe the data itself: which points belong together, how information can be summarized, and which directions carry the most variation. These techniques are invaluable whenever labels are expensive, unavailable, or simply not the point — for example, when exploring a new dataset, compressing high-dimensional signals, or finding anomalies.

Typical unsupervised tasks fall into a few broad categories:

- **Clustering** — grouping observations so that points within a group are more similar to one another than to points in other groups.
- **Dimensionality reduction** — projecting high-dimensional data onto a smaller set of informative axes so the data is easier to visualize, store, or feed into downstream models.
- **Matrix factorization** — decomposing a data matrix into simpler building blocks that expose its latent structure.
- **Semi-supervised learning** — leveraging a small amount of labeled data together with a much larger pool of unlabeled data, bridging supervised and unsupervised settings.

This folder contains self-contained demonstrations of the unsupervised algorithms implemented in `mlpackage`, each in its own subfolder with a short README and a Jupyter notebook. The algorithms covered here are:

- **DBSCAN** — a density-based clustering algorithm that discovers clusters of arbitrary shape and explicitly labels outliers as noise.
- **K-means** — a centroid-based clustering algorithm that partitions the data into K spherical groups by iteratively refining cluster means.
- **Label Propagation** — a graph-based semi-supervised method that spreads labels from a few known points to the rest of the dataset through a similarity graph.
- **PCA** (Principal Component Analysis) — a linear dimensionality-reduction technique that finds the orthogonal directions of greatest variance in the data.
- **SVD** (Singular Value Decomposition) — a foundational matrix factorization underlying PCA, latent semantic analysis, recommender systems, and many other methods.

Each subfolder explains the mathematics behind its algorithm, walks through a worked example, and, where appropriate, compares our from-scratch implementation against a reference library implementation (such as scikit-learn).
